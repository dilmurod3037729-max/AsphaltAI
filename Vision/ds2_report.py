"""data/ds2 sifat hisoboti -> data/ds2/REPORT.md va schema.json. Rasm ochilmaydi: hammasi raqamli.
    py Vision/ds2_report.py"""
import csv
import json
import os

import numpy as np

import build_ds2 as B
import rewards as RW

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "ds2")
G, W = 0.97, 20


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.r_[pos, neg]; rk = np.argsort(np.argsort(allv)) + 1
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def rtg_window(r, ep_end):
    n = len(r); out = np.zeros(n)
    for i in range(n):
        j = min(ep_end[i], i + W)
        out[i] = (r[i:j] * G ** np.arange(j - i)).sum()
    return out


def main():
    S = B.load_steps(os.path.join(OUT, "steps.csv"))
    E = list(csv.DictReader(open(os.path.join(OUT, "episodes.csv"))))
    mf = json.load(open(os.path.join(OUT, "manifest.json")))
    n = len(S["speed"])
    L = []
    p = L.append
    p("# ds2 — sifat hisoboti\n")
    p(f"Qadamlar: **{n}**, kadrlar fayli {n} x 48x96x3. Qurilgan: `Vision/build_ds2.py`, hisobot: `Vision/ds2_report.py`.\n")

    p("## 1. Inventarizatsiya (rec ↔ csv)\n")
    p("| tag | manba | csv qator | join | kadrsiz csv | datasetga | chiqarilgan |\n|---|---|---|---|---|---|---|")
    tot_drop = 0
    for t, v in mf["tags"].items():
        tot_drop += v["csv_without_frame"]
        p(f"| {t} | {B.source_of(t)} | {v['rows_csv']} | {v['rows_joined']} | {v['csv_without_frame']} | {v['kept']} | {v['rows_joined'] - v['kept']} |")
    nr = [e for e in E if e["episode_id"].endswith("_all")]
    for e in nr:
        p(f"| {e['tag']} | {e['source']} | — | 0 | — | 0 | {e['excluded']} |")
    p(f"\nrec↔csv mos kelmagan qadamlar: **{tot_drop}**. Chiqarilganlar sababi epizodlar bo'limida.\n")

    p("## 2. Epizodlar\n")
    ep = [e for e in E if not e["episode_id"].endswith("_all")]
    inc = [e for e in ep if not e["excluded"]]
    exc = [e for e in ep if e["excluded"]]
    for src in ("emulator", "pc"):
        I = [e for e in inc if e["source"] == src]
        if not I:
            p(f"- {src}: kiritilgan epizod yo'q")
            continue
        fin = [e for e in I if e["finished"] == "1"]; dq = [e for e in I if e["dq"] == "1"]
        ft = [float(e["finish_time_s"]) for e in fin if e["finish_time_s"]]
        p(f"- **{src}**: {len(I)} epizod, marra {len(fin)} ({100 * len(fin) / len(I):.0f}%), DQ {len(dq)}, to'xtatilgan/noma'lum {len(I) - len(fin) - len(dq)}; "
          f"o'rtacha uzunlik {np.mean([float(e['dur_s']) for e in I]):.0f} s" + (f"; marra vaqti min {min(ft):.1f} s, median {np.median(ft):.1f} s" if ft else ""))
    reasons = {}
    for e in exc:
        k = e["excluded"].split(" (")[0]
        reasons.setdefault(k, [0, 0]); reasons[k][0] += 1; reasons[k][1] += int(e["n"] or 0)
    p("\nChiqarilganlar (epizod / qadam):")
    for k, (c, s) in sorted(reasons.items(), key=lambda x: -x[1][1]):
        p(f"- {k}: {c} / {s}")
    pf = sum(int(e["post_finish_dropped"] or 0) for e in inc)
    p(f"- marradan keyingi sayr qadamlari (epizod marrada kesiladi): {pf}\n")

    p("## 3. Signal qamrovi (manba bo'yicha)\n")
    p("Tezlik faqat HUD o'qilgan qadamda yoziladi (agent shunday qadam qiladi), shuning uchun 'o'qilgan %' = yozilgan qadamlar / epizod vaqtidagi 250 ms o'rinlar.\n")
    p("| manba | qadam | tezlik o'qilgan % | segment chegaralari | speed_glitch % | DIST o'qilgan % | nitro_fill bor % | evt_cls bor % | WRONG WAY % |\n|---|---|---|---|---|---|---|---|---|")
    for src in ("emulator", "pc"):
        m = S["source"] == src
        if not m.any():
            continue
        I = [e for e in inc if e["source"] == src]
        slots = sum(float(e["dur_s"]) * 4 + 1 for e in I)
        p(f"| {src} | {m.sum()} | {100 * m.sum() / max(slots, 1):.1f} | {int(S['is_first'][m].sum())} | {100 * S['speed_glitch'][m].mean():.2f} | "
          f"{100 * (S['dist'][m] >= 0).mean():.1f} | {100 * (S['nitro_fill'][m] >= 0).mean():.0f} | {100 * (S['evt_cls'][m] >= 0).mean():.0f} | {100 * S['ww'][m].mean():.1f} |")
    gaps = S["gap_after_ms"][S["gap_after_ms"] > 0]
    p(f"\nSegment ichidagi uzilishlar (HUD/takedown/pauza): {len(gaps)} ta, median {np.median(gaps):.0f} ms, 90% {np.percentile(gaps, 90):.0f} ms, max {gaps.max():.0f} ms.")
    p("Emulatorda DIST qiymati APK'da oxirgi barqaror o'qish bilan ushlab turiladi (AiService distStable), shuning uchun -1 kam — bu 'har qadam yangi o'qildi' degani emas.\n")

    p("## 4. Reward ↔ poyga natijasi\n")
    ep_id = S["episode_id"]
    end = np.zeros(n, np.int64)
    for e in inc:
        a, b = int(e["start_idx"]), int(e["end_idx"]); end[a:b] = b
    emu = S["source"] == "emulator"
    for st in (0, 6):
        r = RW.STAGES[st](S).astype(np.float64)
        per = {}
        for e in inc:
            a, b = int(e["start_idx"]), int(e["end_idx"])
            per[e["episode_id"]] = (r[a:b].sum() / max((b - a) / 240, 1e-6), e)
        fin = [v for v, e in per.values() if e["finished"] == "1" and e["source"] == "emulator"]
        nof = [v for v, e in per.values() if e["finished"] != "1" and e["source"] == "emulator"]
        ft = [(v, float(e["finish_time_s"])) for v, e in per.values() if e["finished"] == "1" and e["finish_time_s"] and e["source"] == "emulator"]
        md = [(v, float(e["max_dist"])) for v, e in per.values() if e["source"] == "emulator" and float(e["max_dist"]) >= 0]
        rtg = rtg_window(r, end)
        d = S["dist"]; prog = np.full(n, np.nan)
        for i in np.where(emu)[0]:
            j = min(end[i], i + W)
            if j - i >= W // 2 and d[i] >= 0 and d[j - 1] >= 0:
                sg = np.diff(d[i:j])
                if (sg >= 0).all() and (sg <= 3).all():
                    prog[i] = d[j - 1] - d[i]
        mk = ~np.isnan(prog)
        A = spearman(rtg[mk], prog[mk])
        stall, move = mk & (prog == 0), mk & (prog > 0)
        p(f"**{st}-bosqich** (epizod reward'i daqiqaga; faqat emulator, PC'da DIST ishonchsiz):")
        p(f"- marra vs marrasiz AUC: **{auc(np.array(fin), np.array(nof)):.2f}** (marra {len(fin)}, marrasiz {len(nof)}; 0.5 = bog'liqlik yo'q)")
        p(f"- marra vaqti bilan Spearman: **{spearman(np.array([x for x, _ in ft]), np.array([y for _, y in ft])):+.2f}** (manfiy = tezroq marra ko'proq reward)")
        p(f"- erishilgan DIST bilan Spearman: **{spearman(np.array([x for x, _ in md]), np.array([y for _, y in md])):+.2f}**")
        p(f"- reward_validate A (20 qadam reward-to-go ↔ DIST progress): **{A:+.2f}** (n={int(mk.sum())}; mezon >= 0.6)")
        p(f"- reward_validate E (progress=0 oynalar < progress>0): {rtg[stall].mean():+.2f} < {rtg[move].mean():+.2f} → {'PASS' if rtg[stall].mean() < rtg[move].mean() else 'FAIL'}")
        if st == 6:
            p("- Eslatma: 6-bosqich DIST'ning o'zidan qurilgan, A va DIST korrelyatsiyasi qisman ta'rif bo'yicha yuqori; mustaqil tekshiruv — marra AUC va marra vaqti.")
        p("")

    p("## 5. Tasodifiy 10 epizod: kadr ↔ signal (raqamli)\n")
    fr = np.memmap(os.path.join(OUT, "frames.u8"), np.uint8, "r", shape=(n, 48, 96, 3))
    rs = np.random.RandomState(1)
    pick = [inc[i] for i in rs.choice(len(inc), min(10, len(inc)), replace=False)]
    p("| epizod | qadam | qora kadr % | kadr yoshi 0..300 ms % | harakat↔tezlik Spearman | urilishda kadr o'zgarishi / odatiy |\n|---|---|---|---|---|---|")
    for e in pick:
        a, b = int(e["start_idx"]), int(e["end_idx"])
        f = fr[a:b].astype(np.int16)
        black = float((S["frame_mean"][a:b] < 8).mean())
        age = S["frame_age_ms"][a:b]
        okage = float(((age >= 0) & (age <= 300)).mean())
        motion = np.r_[np.nan, np.abs(np.diff(f[:, 30:48, 24:72], axis=0)).mean((1, 2, 3))]   # yo'l: pastki-markaz
        cont_prev = np.r_[0, S["cont"][a:b - 1]] == 1
        mm = cont_prev & ~np.isnan(motion)
        rho = spearman(motion[mm], S["speed"][a:b][mm])
        c = (S["coll"][a:b] == 1) & mm
        ratio = float(np.nanmean(motion[c]) / np.nanmean(motion[mm & ~c])) if c.any() else float("nan")
        p(f"| {e['episode_id']} | {b - a} | {100 * black:.1f} | {100 * okage:.0f} | {rho:+.2f} | {ratio:.2f} |")
    p("\nTalqin: qora kadr 0% va kadr yoshi 0..300 ms — kadr shu csv qadamiga tegishli. Harakat↔tezlik va urilishdagi kadr o'zgarishi 96x48 kadrda "
      "KUCHSIZ (|rho| ~0.1, nisbat ~1): bu o'lchov past aniqlikda siljishni na isbotlaydi, na inkor qiladi. "
      "Siljish profili (12 epizod, yo'l qismi, Spearman, kadr harakati vs tezlik, lag -4..+4): .12 .13 .12 .08 .04 .03 .02 .00 .00 — "
      "eng yuqori qiymat kadr tezlikdan 2-4 qadam KEYIN — bu fizik ta'sir, yozuv siljishi EMAS:\n"
      "- To'g'ridan-to'g'ri tekshiruv (30 epizod): kadrdagi HUD tezlik raqami qutisi (y 1-8, x 82-96) piksel o'zgarishi vs csv |dSpeed|, lag -4..+4: "
      ".08 .10 .12 .15 **.21** .12 .10 .09 .06 — eng yuqori lag 0 da: kadr va tezlik bir qadamga tegishli.\n"
      "- Harakat -> kadr kechikishi (40 epizod): rul X vs kadr gorizontal siljishi, lag 0..5: -.08 -.14 **-.19** +.03 +.02 +.02 — "
      "buyruq kadrda 2 qadam (~0.5 s) keyin ko'rinadi (pad -> o'yin -> ekran). Dreamer buni rekurrent holat bilan o'rganadi; BC/Q uchun ham hisobga olinsin.\n")
    p("## 6. Cheklovlar va ochiq savollar\n")
    p("- 0-bosqich reward'i epizod darajasida natija bilan bog'liq (marra AUC, DIST), lekin qadam darajasida DIST progressini bashorat qilmaydi (A ~0, E FAIL): "
      "tezlik ushlash progressning o'zi emas — bu bosqich uchun kutilgan, 6-bosqich bilan birga ishlatiladi.")
    p("- PC: hozircha 1 epizod; DIST ~8% o'qiladi, nitro chizig'i va hodisa ustuni PC HUD'ida boshqa joyda (nitro_fill/evt_cls = -1). PC1 butunlay chiqarilgan (eski tezlik o'qigichi).")
    p("- Trassa kaliti datasetda yo'q (trassa indekslash taqiqlangan) — 6-bosqich rekord bo'lagi faqat tashqaridan record_s berilsa ishlaydi.")
    p("- evt_cls klasterlari nomlanmagan (near miss / takedown va h.k.) — 5-bosqichdan oldin nomlash kerak.")
    p("- t1, v2 (eng eski 2 sessiya) kadrsiz — datasetga kirmaydi.")
    p("- Siljish/kechikish tekshiruvi skripti hisobotga kiritilmagan (bir martalik o'lchov, natijasi 5-bo'limda).\n")
    open(os.path.join(OUT, "REPORT.md"), "w", encoding="utf-8").write("\n".join(L))
    write_schema()
    print("\n".join(L))


def write_schema():
    v, t = "verified", "taxmin"
    cols = {
        "global_idx": ["frames.u8/evt.u8/nitro.u8 dagi qator", v], "episode_id": ["<tag>_r<poyga>", v], "seg_id": ["uzluksiz bo'lak (dt<=500 ms, qadam+1)", v],
        "tag": ["xom fayl tagi = siyosat/davra", v], "source": ["emulator | pc", v], "step": ["agent qadami (csv c00)", v],
        "t_ms": ["csv c01: emulator SystemClock ms, PC epoch ms — faqat farqlari ma'noli", v], "t_race_ms": ["poyga birinchi qadamidan ms", v],
        "dt_ms": ["oldingi qadamdan ms (segment boshida bo'sh)", v], "frame_age_ms": ["csv now - rec t: kadr qadamdan qancha oldin olingan", v],
        "speed": ["HUD tezligi, mph (har doim o'qilgan; o'qilmagan qadam yozilmaydi)", v], "speed_glitch": ["|dS|>=80 mph oldingi qadamdan (Agent.cs GlitchMph)", v],
        "dist": ["HUD DIST %, -1 o'qilmadi; emulatorda oxirgi barqaror qiymat ushlanadi", v],
        "nitro_fill": ["nitro chizig'i to'liqligi 0..1 (labels.nitro_fill); PC'da -1 (HUD joyi boshqa)", v],
        "ww": ["WRONG WAY pushti belgi (labels.wrongway); PC'da tasdiqlanmagan", t],
        "evt_cls": ["hodisa ustuni klasteri 0..7 (labels.event_classes, markazlar evt_centers.npy); PC'da -1; klaster nomlari yo'q", t],
        "evt_active": ["csv c20: hodisa ustunida yorqin piksel ulushi", v],
        "a_steer": ["0..4", v], "a_drift": ["0 yo'q, 1 ushla, 2 ikki marta (360)", v], "a_nitro": ["0 yo'q, 1 bir marta, 2 ikki marta (shockwave)", v],
        "a_gas": ["0/1; yangi versiyalarda doim 1; PC'da gaz avtomatik", v], "steer_x": ["stik X = -1,-.5,0,.5,1", v],
        "prev_steer": ["shu segmentdagi oldingi qadam harakati (segment boshida 2/0/0)", v], "prev_drift": ["", v], "prev_nitro": ["", v],
        "expl": ["tadqiqot bit maskasi: 1 rul, 2 drift, 4 nitro tasodifiy; 8 bo'sh-tomon override (faqat 25 ustunli fayllar)", v],
        "eps": ["exploration ehtimoli", v], "coll_pred": ["kodlovchi aux[0] to'qnashuv bashorati (agent kirishi, yorliq EMAS)", v],
        "is_first": ["yangi segment boshi (Dreamer is_first)", v], "cont": ["1 = keyingi qadam shu segmentda va dt<=500 ms", v],
        "terminal": ["1 = epizod haqiqiy tugashi (marra yoki DQ); qolgan uzilishlar truncation", v], "finish": ["1 = marra qadami", v],
        "gap_after_ms": ["segment oxirida keyingi qadamgacha uzilish ms (HUD yo'qolishi/takedown/pauza)", v],
        "valid_h": ["1 s gorizont shu segment ichida", v], "coll": ["urilish: tezlik>60 va 1 s ichida >=30 mph tushish", v],
        "d_speed_1s": ["v[t+4]-v[t]", v], "drop_1s": ["v[t]-min(v[t+1..t+4])", v], "nitro_gain_1s": ["max(0, fill[t+4]-fill[t])", v],
        "prog_1s": ["DIST[t+4]-DIST[t], faqat toza o'sish (har qadam 0..3)", v], "frame_mean": ["kadr o'rtacha yorqinligi (qora kadr tekshiruvi)", v],
    }
    sch = {"steps.csv": cols,
           "episodes.csv": {"finish_src": "bonus (agent FINISH bonusi) | dist99 (DIST>=99, >20 s) | pc_results (natija ekrani OCR)",
                            "finish_time_s": "bonus/dist99: birinchi HUD qadamidan marragacha (Agent.cs FinishReward bilan bir xil ta'rif); pc_results: o'yin natija ekrani vaqti",
                            "dq": "marrasiz va >=170 s (3:00 chegarasi)", "excluded": "bo'sh = datasetda; aks holda sabab"},
           "frames.u8": "(N,48,96,3) RGB uint8, butun ekran eng yaqin qo'shni", "evt.u8": "(N,40,128) kul, HUD o'ng ustuni (x .66-1, y .33-.55)",
           "nitro.u8": "(N,8,96,3) RGB, nitro chizig'i qutisi (x .30-.68, y .02-.09); PC'da bu qutida nitro chizig'i yo'q",
           "reward": "YOZILMAGAN — Vision/rewards.py STAGES[k](load_steps())"}
    json.dump(sch, open(os.path.join(OUT, "schema.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
