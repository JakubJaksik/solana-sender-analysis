"""
Cross-check lost triggers vs leader / country / timing.
Adds intended-slot vs next-slot vs later landing analysis.

Usage:
  python analyze_lost.py <triggers.jsonl> <validators-epoch-N.json>
"""
import json
import sys
from collections import defaultdict, Counter

triggers_path, vfile = sys.argv[1], sys.argv[2]

# 1. slot → leader_identity from schedule
vj = json.load(open(vfile))
epoch_first_slot = vj["epoch"]["absolute_slot"]
slot_to_leader = {}
for identity, slot_indices in vj["schedule"].items():
    for si in slot_indices:
        slot_to_leader[epoch_first_slot + si] = identity

def parse_country(dc):
    if not dc or not isinstance(dc, str): return None
    p = dc.split("-")
    return p[1].upper() if len(p) >= 2 and len(p[1]) == 2 else None

id_info = {}
for v in vj["validators"]:
    id_info[v["identity"]] = {
        "name": v.get("name") or "(unnamed)",
        "country": parse_country(v.get("data_center_key")),
        "dc": v.get("data_center_key"),
    }

# 2. aggregate triggers.jsonl per trigger_id
per_trigger = defaultdict(lambda: {"slot": None, "tick": None, "attempts": []})
with open(triggers_path) as f:
    for line in f:
        r = json.loads(line)
        t = per_trigger[r["trigger_id"]]
        t["slot"] = r["slot"]
        t["tick"] = r["tick"]
        t["attempts"].append({
            "sender_id": r["sender_id"],
            "outcome": r["final_outcome"],
            "observed_slot": r.get("observed_slot"),
            "observed_tick": r.get("observed_tick"),
            "send_to_obs_ns": r.get("wall_send_to_observed_ns"),
        })

# 3. classify
landed_by_leader = Counter()
lost_by_leader = Counter()
intended_hit_by_leader = Counter()
landed_by_country = Counter()
lost_by_country = Counter()
intended_hit_by_country = Counter()

slot_delta_hist = Counter()
landed_send_to_obs = []
lost_detail = []
intended_misses_detail = []

for tid, t in per_trigger.items():
    leader = slot_to_leader.get(t["slot"], "UNKNOWN")
    info = id_info.get(leader, {"name": "??", "country": None, "dc": None})

    landed_att = [a for a in t["attempts"] if a["outcome"] == "Landed"]
    is_landed = len(landed_att) > 0
    if is_landed:
        winning = landed_att[0]
        delta = (winning["observed_slot"] or t["slot"]) - t["slot"]
        slot_delta_hist[delta] += 1
        landed_by_leader[leader] += 1
        landed_by_country[info["country"] or "??"] += 1
        if delta == 0:
            intended_hit_by_leader[leader] += 1
            intended_hit_by_country[info["country"] or "??"] += 1
        else:
            actual_leader = slot_to_leader.get(winning["observed_slot"], "UNKNOWN")
            actual_info = id_info.get(actual_leader, {})
            intended_misses_detail.append({
                "slot": t["slot"], "tick": t["tick"], "delta": delta,
                "intended_name": info["name"], "intended_cc": info["country"],
                "actual_name": actual_info.get("name", "?"),
                "actual_cc": actual_info.get("country"),
            })
        if winning["send_to_obs_ns"]:
            landed_send_to_obs.append(winning["send_to_obs_ns"] / 1e6)
    else:
        lost_by_leader[leader] += 1
        lost_by_country[info["country"] or "??"] += 1
        lost_detail.append({
            "slot": t["slot"], "tick": t["tick"],
            "name": info["name"], "country": info["country"], "dc": info["dc"],
        })

total = len(per_trigger)
landed_total = sum(landed_by_leader.values())
intended_total = sum(intended_hit_by_leader.values())
lost_total = sum(lost_by_leader.values())

print("=== OVERVIEW ===")
print(f"triggers total                  : {total}")
print(f"  landed (anywhere)             : {landed_total} ({landed_total/total*100:.1f}%)")
print(f"  landed on INTENDED slot       : {intended_total} ({intended_total/total*100:.1f}%)")
print(f"  landed on later slot          : {landed_total-intended_total} ({(landed_total-intended_total)/total*100:.1f}%)")
print(f"  lost (no land)                : {lost_total} ({lost_total/total*100:.1f}%)")
print()

print("=== Slot delta distribution (observed_slot - intended_slot) ===")
for d in sorted(slot_delta_hist):
    n = slot_delta_hist[d]
    bar = "#" * min(60, n // 2 + 1)
    print(f"  delta={d:+3d}  n={n:4d}  {bar}")
print()

print("=== LOST + INTENDED-MISS - by leader (top 25 by 'not_intended') ===")
print(f"{'lost':>5} {'miss':>5} {'land':>5} {'hit%':>6}  {'cc':<3} {'name':<35} {'dc':<28}")
all_leaders = set(landed_by_leader) | set(lost_by_leader)
def not_intended(i):
    return lost_by_leader[i] + (landed_by_leader[i] - intended_hit_by_leader[i])
for ident in sorted(all_leaders, key=not_intended, reverse=True)[:25]:
    lost = lost_by_leader[ident]
    land = landed_by_leader[ident]
    intended = intended_hit_by_leader[ident]
    miss = land - intended
    total_l = lost + land
    info = id_info.get(ident, {})
    hit_pct = intended / total_l * 100 if total_l else 0
    print(f"{lost:>5} {miss:>5} {land:>5} {hit_pct:>5.0f}%  {info.get('country') or '??':<3} {(info.get('name') or '?')[:35]:<35} {(info.get('dc') or '?')[:28]:<28}")
print()

print("=== Lost+miss by COUNTRY ===")
print(f"{'cc':<3} {'lost':>5} {'miss':>5} {'land':>5} {'hit%':>6}")
all_cc = set(landed_by_country) | set(lost_by_country)
def cc_not_intended(c):
    return lost_by_country[c] + (landed_by_country[c] - intended_hit_by_country[c])
for cc in sorted(all_cc, key=cc_not_intended, reverse=True):
    lost = lost_by_country[cc]
    land = landed_by_country[cc]
    intended = intended_hit_by_country[cc]
    miss = land - intended
    total_c = lost + land
    hit_pct = intended / total_c * 100 if total_c else 0
    print(f"{cc:<3} {lost:>5} {miss:>5} {land:>5} {hit_pct:>5.0f}%")
print()

print("=== Landed send→observed latency (ms) ===")
if landed_send_to_obs:
    s = sorted(landed_send_to_obs)
    p50 = s[len(s)//2]; p95 = s[int(len(s)*0.95)]; p99 = s[int(len(s)*0.99)]
    print(f"n={len(s)}  min={s[0]:.0f}  p50={p50:.0f}  p95={p95:.0f}  p99={p99:.0f}  max={s[-1]:.0f}")
print()

print("=== Intended-slot misses (landed but on later slot) - top 30 ===")
for x in sorted(intended_misses_detail, key=lambda r: -r["delta"])[:30]:
    print(f"  slot={x['slot']} tick={x['tick']:2d} delta=+{x['delta']}  intended={x['intended_name'][:20]:<20} ({x['intended_cc'] or '??'}) → actual={x['actual_name'][:20]:<20} ({x['actual_cc'] or '??'})")
print()

print("=== Lost triggers - full detail ===")
for x in sorted(lost_detail, key=lambda r: r["slot"]):
    print(f"  slot={x['slot']} tick={x['tick']:2d}  leader={(x['name'] or '?')[:25]:<25} cc={x['country'] or '??'}  dc={x['dc'] or '?'}")
