# M6 — Analysis & Thesis Writing — Sessions 6+

> **Goal ek line mein:** Results ko RQ1–RQ3 ke jawabon mein badalna, thesis document likhna, final defense deck, aur repo ki tagged release. Ye milestone coding kam, likhna zyada hai — Claude Code yahan editor/formatter hai, author aap hain.

## Definition of Done

- [ ] RQ1/RQ2/RQ3 har ek ka ek-paragraph verdict, evidence (table/figure) ke hawale se
- [ ] Thesis chapters draft: Intro, Background, Related Work, Methodology, Experiments, Results & Discussion, Conclusion & Future Work
- [ ] Saare figures 300-DPI PDF, numbered, captioned; saare tables final numbers ke saath
- [ ] Limitations section **imandaari se**: capped samples, R=10, ek task, ek model, 2 seeds
- [ ] AI-tool disclosure acknowledgements mein (pehle agreed wording)
- [ ] Repo: final README (Results section ab add ho), `v1.0-thesis` git tag, reproduce-instructions verified ek fresh clone se
- [ ] Final defense deck: topic-defense v3 ko update karke "Reference Results" ki jagah AAPKE results
- [ ] Sir se chapter-by-chapter feedback loop (ek saath sab kuch akhri hafte mat bhejein)

## Likhne Ki Tarteeb (mera mashwara — Intro pehle NAHI)

1. **Methodology** (sab se asaan — code ho chuka, M2–M5 files hi outline hain)
2. **Experiments & Results** (tables/figures taiyar hain)
3. **Related Work** (defense deck ke 8+1 papers ka expansion)
4. **Discussion** (RQ verdicts + limitations + surprises)
5. **Background & Intro** (ab likhna asaan — aap ko pata hai kahani kahan pohanchi)
6. **Conclusion & Future Work** (scoped-out cheezein: privacy, heterogeneity, bare models, testbed)
7. **Abstract akhir mein**

## Part A: Aap Khud

- University ka thesis template (Word/LaTeX) le lein — format wahan se aata hai
- Sir se chapter deadline schedule bana lein (haftawar)

## Part B: Claude Code Prompts (choti sessions, zaroorat ke mutabiq)

```
Session 6a — Results consolidation
TASK: scripts/make_thesis_assets.py — read all results/*.json, regenerate
every figure (300 DPI PDF) and every table (both markdown and LaTeX tabular)
into thesis_assets/. Single command = all assets reproducible.
Then: verify numbers in tables match source JSONs (print a checksum-style
summary I can eyeball).

Session 6b — Repo finalization
TASK 1: Update README.md — add Results section (headline numbers + 2 figures),
finalize reproduce instructions; test them mentally against a fresh
`git clone --recurse-submodules`.
TASK 2: Propose `git tag v1.0-thesis` + commands (I run).

Session 6c — Writing support (per chapter, as needed)
I will paste my drafted chapter text. Your job: grammar, clarity, academic
tone, consistency of symbols (K, B, R, S), citation-number consistency
[1]-[9]. DO NOT add claims, numbers, or references I didn't write. Flag
anything that sounds overclaimed — I want honest text.
```

**Zaroori usool:** thesis ka **matan aap likhein**, Claude Code sirf polish kare. Ye academic integrity ka masla bhi hai aur defense ka bhi — examiner ke sawalon ka jawab wohi de sakta hai jisne likha ho.

## Part C: Final Deck Update

Mujhe (Claude chat) M5 ka `summary.json` + plots dein — main topic-defense v3 ko final-defense deck mein update kar doonga: Reference-Results slide ko aap ke asal GLUE results se badal kar, aur script notes bhi.

## Common Problems

| Problem | Hal |
|---|---|
| "Results kamzor lag rahe" | Framing dekhein: same accuracy at less comm = jeet; mixed = analyze kahan/kyun — M5 ka honest-framing note |
| Numbers tables mein mismatch | sirf make_thesis_assets.py se regenerate — haath se kabhi copy na karein |
| Writing block | M2–M5 ki .md files kholein — har file methodology ka ready outline hai |
| Deadline pressure | Chapters ki tarteeb upar wali rakhein — Methodology+Results pehle done = 60% thesis done |
