def sub(p, a, b):
    s = open(p, encoding='utf-8').read()
    assert a in s, (p, a[:40])
    open(p, 'w', encoding='utf-8').write(s.replace(a, b, 1))
sub('sections/setup.tex',
    "was trained forward-only under the identical gate and recipe\n(reaching \minADESixWayformer, Sec.~\ref{sec:wf})",
    "was trained forward-only under the same accuracy gate and effective batch size, with its own\nlearning rate and 27-epoch schedule (reaching \minADESixWayformer, Sec.~\ref{sec:wf})")
sub('sections/intro.tex',
    "confirmed on\ntwo architectures matching published accuracy;",
    "confirmed on\ntwo architectures within 15\% of published accuracy;")
sub('sections/intro.tex',
    "and controlled input perturbations localise the loss to turns in left-hand traffic.",
    "the loss concentrates in turns in left-hand traffic, and controlled input perturbations account for part of it.")
