# Vision — o'rganiladigan ko'rish

Kirish: ketma-ket 3 kadr (9 kanal, 48x96). Chiqish: embedding (128) + yordamchi bosh'lar. Hammasi o'yinning o'z signalidan o'rganiladi:

| Bosh | Yorliq (labels.py) | Nima o'rgatadi |
|---|---|---|
| to'qnashuv | 1 s ichida tezlik >= 20 mph tushdimi | devor/mashina/to'siqni vizual his qilish |
| tezlik yo'qotishi, o'zgarishi | kelajak tezlik | yo'l va holatni baholash |
| nitro o'sishi | nitro chizig'i to'lishi | o'yin "yaxshi haydash" deb bilgan vaziyatlar |
| hodisa sinfi (8) | HUD o'ng ustuni (DRIFT, AIR TIME, ...) klasteri | hodisalarni tanish (`events_sheet.py` bilan nomlanadi) |
| kelajak reward | diskontlangan reward yig'indisi | qiymat oldindan o'rgatish |

Fayllar: `data.py` (rec.bin o'qish), `labels.py`, `model.py`, `train.py` (`--export` ONNX'ni APK'ga yozadi), `evaluate.py` (faqat-tezlik bilan taqqoslash AUC + saliency rasmlari), `events_sheet.py`, `reward_audit.py`.
Ma'lumot: `../data/rec_<tag>.bin` + `agent_<tag>.csv` juftliklari; tekshiruv — oxirgi 15% epizodlar.
