# Vazifa: AsphaltAI uchun yagona, kuchli dataset (ds2) tayyorlash

Loyiha: `V:\Projects\AsphaltAI`. Avval xotiradagi `project_asphaltai.md` ni va shu faylni to'liq o'qi.

## Nega kerak
Keyingi ikki ish shu datasetga tayanadi:
1. **Curriculum (bosqichli o'rgatish)** — AI'ga ko'nikmalar bitta-bitta beriladi: 0) tezlikni ushlash (keraksiz drift/tormoz jazosi), 1) nitro, 2) tezlikni ushlab rul burish (zigzag/aldash jazosi), 3) urilish = "u yerda yo'l yo'q", 4) burilishda drift, 5) triklar (nitro to'lishi, near miss), 6) natija (progress, marra, rekordga nisbatan bo'lak vaqti). Har bosqichning reward'i ESKI poygalarda ham qayta hisoblanishi kerak.
2. **DreamerV3 world model** (PyTorch, fp32, ~50M, RTX 2080 Super 8 GB) — oflayn shu ma'lumotda o'qitiladi: kadr + harakat + continue flag + reward.

Hozirgi holat ikkalasiga yaroqsiz:
- 25 ta tarqoq juftlik `data/rec_<tag>.bin` + `data/agent_<tag>.csv`, jami ~168k qadam (~3.5 GB), yagona indeks yo'q.
- `agent_*.csv` dagi reward ustuni (c07) formulasi fayldan faylga FARQ qiladi — ishlatib bo'lmaydi.
- PC yozuvlari (PC1, PC2) formati boshqa: vaqt epoch ms (emulatorda SystemClock), ba'zi ustunlar ma'nosi farq qilishi mumkin.
- Eski eksport `data/dataset/` (2026-10-20) atigi 7 epizod (~5k qadam).
- Epizod chegaralari, takedown sahnalari, HUD uzilishlari, pauza bir xil belgilanmagan.

## Manbalar (avval o'qi, taxmin qilma)
- `Vision/data.py` — REC dtype: int32 step, int64 t(ms), frame 48x96x3 RGB, evt 40x128 kul (HUD o'ng ustuni), nitro 8x96x3. Qadam 250 ms (4 Hz).
- `data/dataset/schema.json` — agent csv ustunlari ma'nosi (sarlavhasiz; eski fayllarda faqat 11 ustun). Shubhali ustunni `App/Agent.cs` (Loop, Reward ~229-qator) va `tools/pc_agent.py` dan tekshir.
- `Vision/export_dataset.py`, `Vision/labels.py`, `Vision/split_events.py` — mavjud epizod ajratish va yorliq mantiqi; QAYTA ISHLAT va kengaytir, noldan yozma.
- `Vision/reward_validate.py` (mezonlar A–E), `Vision/reward_audit.py`.
- `data/pc_races.csv`, `data/autoloop.json`, `data/ends_<tag>/` (DQ dalillari), `data/audit_*.md`.
- `tools/dist_templates.py` — DIST shablonlari (PC'da DIST hozir ~8% o'qiladi; bu vazifa DIST'ni TUZATMAYDI, faqat qamrovini o'lchaydi).

## Nima qilish kerak
1. **Inventarizatsiya**: har tag uchun qadam soni, rec↔csv qadam mosligi (step bo'yicha join; mos kelmaganlar soni), manba (emulator/PC), siyosat versiyasi (tag nomidan: a1, L1, R2-6, D13..K14, PC1-2 va h.k.).
2. **Epizodlarga ajratish**: poyga boshlanishi/tugashi, marra (DIST>=99 yoki natija), DQ (~3:00 vaqt chegarasi), qo'lda to'xtatilgan. Juda qisqa (<20 s) yoki singan epizodlarni CHIQAR, lekin sababini ro'yxatga yoz.
3. **Uzilishlar → continue flag**: HUD tezligi yo'qolgan oraliqlar (takedown sahnasi, kamera), pauza, kadr vaqtidagi sakrashlar (>2× qadam). Dreamer uchun segment chegarasi sifatida belgilansin.
4. **Xom signallar bir xil birlikda** (reward EMAS): tezlik (mph, o'qilmasa -1), DIST % (o'qilmasa -1), nitro chizig'i darajasi, WRONG WAY, hodisa ustuni sinfi (labels.py), harakatlar (steer 0-4, drift 0-2, nitro 0-2), oldingi harakat, dt.
5. **Hosila yorliqlar** (labels.py mantiqi bilan): urilish (1 s da tezlik tushishi; >60 mph dan -30 mph), tezlik o'zgarishi, nitro o'sishi, progress tezligi.
6. **Bosqich reward funksiyalari**: `Vision/rewards.py` — har curriculum bosqichi uchun SOF funksiya `reward_stageN(steps_df) -> np.ndarray`, faqat xom signallardan. Hozircha 0-bosqich (tezlik ushlash, keraksiz drift jazosi) va 6-bosqich (progress/marra) to'liq; qolganlari skelet + izoh.
7. **Saqlash** `data/ds2/`:
   - `frames.u8` — barcha kadrlar memmap (N,48,96,3), `evt.u8`, `nitro.u8` shunga o'xshash;
   - `steps.parquet` (yoki csv) — har qator bitta qadam: global_idx, episode_id, t_ms, signallar, harakatlar, cont flag, yorliqlar;
   - `episodes.csv` — episode_id, tag, manba, siyosat, start/end idx, uzunlik s, finished, vaqt, DQ, o'rtacha tezlik, chiqarilgan bo'lsa sabab;
   - `schema.json` — har ustun ma'nosi va manbasi (verified/taxmin);
   - `REPORT.md` — sifat hisoboti (pastda).
   - Yaratuvchi skript: `Vision/build_ds2.py` (qayta ishga tushirilsa bir xil natija; yangi tag qo'shilsa davom ettiradi).
8. **Tekshiruv** (REPORT.md ga raqam bilan):
   - rec↔csv mos kelmagan qadamlar soni = 0 yoki sababi tushuntirilgan;
   - tezlik o'qilgan %, DIST o'qilgan % — manba (emulator/PC) bo'yicha alohida;
   - epizodlar soni, marra %, chiqarilganlar va sabablari;
   - 0- va 6-bosqich reward'i epizod natijasi bilan korrelyatsiyasi (`reward_validate` mezonlari bilan);
   - tasodifiy 10 epizoddan kadr+signal tekshiruvi — rasm OCHMA, raqamli moslik bilan (masalan, tezlik o'qilgan qadamda kadr qora emasligi, urilish yorlig'i atrofida tezlik tushishi).

## Qoidalar
- Xom fayllarni (`rec_*.bin`, `agent_*.csv`) O'ZGARTIRMA va O'CHIRMA — faqat o'qi.
- Ish CPU'da; og'ir skript BelowNormal ustuvorlikda, fonda, logi faylga. GPU kerak bo'lsa avval `nvidia-smi`, umumiy util va VRAM <= 80% (`Vision/gpuguard.py`, `tools/safe_run.py`).
- O'yin yoki emulator bilan ishlama — bu vazifa to'liq oflayn.
- Kelishuvlar: RAM qiymatlari AI kirishiga BERILMAYDI; trassa/xarita indekslash va Gameloft bot yo'li ishlatilmaydi.
- Kodni mavjud uslubda yoz, keraksiz izoh yo'q, o'zbekcha nomlar/izohlar loyihadagidek.
- Bilmagan narsani taxmin qilma: ustun ma'nosi aniq bo'lmasa schema.json da "taxmin" deb belgila va REPORT.md ga yoz.
- Bu vazifa DIST o'qishni tuzatish, Dreamer yoki curriculum trening EMAS — faqat dataset. Tugagach REPORT.md xulosasini qisqa ayt va keyingi qadamni kut.
