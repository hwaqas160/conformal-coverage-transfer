# Paper 01, in plain language

## What this paper is about

Self-driving cars constantly predict what everyone around them will do next — which
lane a car will merge into, whether a pedestrian will step off the curb. This is called
**trajectory prediction**, and it's one of the things a self-driving system's AI model
does dozens of times a second.

Because being wrong can be dangerous, modern systems don't just predict a single path —
they also try to say *how sure* they are, using a technique called **conformal
prediction**. It comes with a real mathematical guarantee: calibrate it properly, and it
promises something like *"the true path will fall inside this zone 90% of the time"* —
provably, not just in theory.

## The problem

That guarantee only holds if the driving situations you calibrate on and the ones you
actually encounter come from the same statistical world. But real systems get
calibrated on data from one place — one city, one sensor setup, one driving culture —
and then deployed somewhere else entirely.

**Nobody has carefully measured what happens to that safety guarantee when a system
crosses that boundary.** It might hold up fine. Or a system might keep confidently
claiming "90% sure" while quietly being right far less often than that — and nobody
would know, because nothing is currently designed to catch it.

## What we're actually doing

1. Train a small trajectory-prediction model (called AutoBot) on a large driving
   dataset collected in Pittsburgh and Miami (Argoverse 2).
2. Calibrate its confidence claims properly, on held-out data from those same cities.
3. Test those same confidence claims on a **completely different** dataset collected in
   Boston and Singapore (nuScenes) — cities, roads, and driving styles the model has
   never seen.
4. Measure exactly how much the guarantee breaks down when it does.
5. Check *why* — do measurable differences (how many cars are nearby, how fast people
   drive, how curvy the roads are) explain the gap?
6. Try to **fix it** — using a method that adjusts the model's confidence based only on
   those measurable differences, without needing any expensive new labeled data from
   the new city.

## Why it matters

- **Safety.** A self-driving system whose stated confidence quietly stops being
  trustworthy the moment it drives somewhere new is a real, currently invisible risk.
- **Nobody has published this measurement.** Plenty of work shows that raw accuracy
  drops in a new city. Nobody has shown whether the *safety guarantee itself* survives.
- **A working fix, not just a warning.** If we can also show a repair that restores the
  guarantee without new labels, that's something the field can use immediately, not just
  a paper pointing out a flaw.

## Target journal

- **Primary: IEEE Robotics and Automation Letters (RA-L).** Fast review, well respected
  in robotics, and normally comes paired with a conference talk (ICRA or IROS) —
  valuable for visibility.
- **Alternative: IEEE Transactions on Intelligent Vehicles.** Higher impact factor,
  more prestige, slower review process.

## The bigger picture

This is the first of three connected papers, built to form a strong research portfolio
for funded PhD applications to autonomous-systems labs in the US and Canada (e.g.
UPenn's Safe Autonomous Systems Lab, Stanford's Autonomous Systems Lab).

## Where things stand right now

- Environment built, Argoverse 2 fully downloaded and converted, nuScenes converted
- A real model is training right now
- The core measurement code is written and tested
- Next: finish training, then run the actual before/after comparison — that produces
  the paper's central number
