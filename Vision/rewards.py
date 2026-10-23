"""Curriculum bosqichlari reward'lari: data/ds2/steps.csv xom signallaridan SOF funksiyalar (yon ta'sirsiz, faqat kirishdan).
    from build_ds2 import load_steps; S = load_steps(); r = reward_stage0(S)
Qoidalar: keyingi qadam qiymati faqat cont==1 bo'lsa ishlatiladi (segment chegarasidan o'tilmaydi); speed_glitch qadamlarida 0.
O'lchov: 4 Hz qadam; og'irliklar App/Agent.cs Reward bilan bir masshtabda (bitta qadam odatda -8..+5)."""
import numpy as np

STABLE_W, STABLE_BAND, UP_W, LOSS_W = 0.5, 1.5, 0.10, 2.0
USELESS_DRIFT_W, NO_GAS_W = 0.2, 0.2
PROG_W, BACK_W, WW_W = 2.0, 3.0, 1.0
FINISH_BONUS, DQ_PENALTY = 20.0, 10.0
R_MIN, R_MAX = -8.0, 5.0


def _next(S, k):
    """k ustunining keyingi qadamdagi qiymati; cont==0 bo'lsa nan."""
    v = S[k].astype(np.float64)
    nx = np.r_[v[1:], np.nan]
    return np.where(S["cont"] == 1, nx, np.nan)


def _ok(S):
    return (S["cont"] == 1) & (S["speed_glitch"] == 0) & (np.r_[S["speed_glitch"][1:], 0] == 0)


def reward_stage0(S):
    """0) Tezlikni ushlash: barqaror yuqori tezlik va tezlanish mukofoti, tezlik yo'qotish (kinetik energiya) jazosi,
    keraksiz drift (to'g'ri rulda drift) va gazni qo'yib yuborish jazosi."""
    v = S["speed"].astype(np.float64); vn = _next(S, "speed"); ok = _ok(S)
    dv = np.nan_to_num(vn - v)
    r = np.where(np.abs(dv) <= STABLE_BAND, STABLE_W * v / 150, 0.0) + UP_W * np.maximum(dv, 0)
    loss = v * v - np.nan_to_num(vn) ** 2
    r -= np.where(loss > 0, LOSS_W * loss / 10000, 0.0)
    r = np.where(ok, r, 0.0)
    r -= USELESS_DRIFT_W * ((S["a_drift"] > 0) & (S["a_steer"] == 2))
    r -= NO_GAS_W * (S["a_gas"] == 0)
    return np.clip(r, R_MIN, R_MAX).astype(np.float32)


def reward_stage1(S):
    """1) Nitro: nitro tugmasi tezlik o'sishiga aylanganmi (a_nitro>0 dan keyin d_speed_1s > 0), nitro chizig'ini behuda sarflash jazosi.
    SKELET: PC'da nitro_fill = -1 (HUD joyi boshqa) — avval PC nitro chizig'i o'qilishi kerak."""
    raise NotImplementedError


def reward_stage2(S):
    """2) Tezlikni ushlab rul burish: rul o'zgarishi tezlik yo'qotmasa mukofot; zigzag (ishora almashishi, prev_steer bilan)
    va "aldash" (rulni ko'p burib tezlikni saqlash — yo'l bo'ylab emas) jazosi. SKELET."""
    raise NotImplementedError


def reward_stage3(S):
    """3) Urilish = "u yerda yo'l yo'q": coll==1 bo'lgan qadam va undan oldingi 1 s dagi rul tanlovi jazolanadi,
    urilishdan keyin bo'sh tomonga burilib tezlikni tiklash mukofotlanadi. SKELET."""
    raise NotImplementedError


def reward_stage4(S):
    """4) Burilishda drift: drift (a_drift==1) rul burilgan holda va keyin tezlik saqlangan bo'lsa mukofot. SKELET."""
    raise NotImplementedError


def reward_stage5(S):
    """5) Triklar: nitro_gain_1s (nitro chizig'i to'lishi), hodisa ustuni (evt_cls: near miss / takedown sinflari). SKELET:
    evt_cls klasterlari nomlanmagan (labels.event_classes), PC'da -1."""
    raise NotImplementedError


def reward_stage6(S, record_s=None):
    """6) Natija: DIST% toza o'sishi (0 < d <= 3 qadamda), orqaga ketish va WRONG WAY jazosi, marra bonusi, DQ jazosi.
    record_s: {episode_id: o'sha trassa rekordi, s} berilsa marrada rekordga nisbatan bo'lak bonusi (trassa kaliti datasetda yo'q)."""
    d = S["dist"].astype(np.float64); dn = _next(S, "dist")
    dd = np.where((d >= 0) & (dn >= 0), dn - d, 0.0)
    dd = np.nan_to_num(dd)
    r = np.where((dd > 0) & (dd <= 3), PROG_W * dd, 0.0)
    r += np.where((dd < 0) & (dd >= -3), BACK_W * dd, 0.0)
    r -= WW_W * (S["ww"] == 1)
    r += FINISH_BONUS * (S["finish"] == 1)
    r -= DQ_PENALTY * ((S["terminal"] == 1) & (S["finish"] == 0))
    if record_s:
        t = S["t_race_ms"] / 1000.0
        for i in np.where(S["finish"] == 1)[0]:
            best = record_s.get(S["episode_id"][i])
            if best:
                r[i] += 10 + 2 * float(np.clip(best - t[i], -20, 20))
    return r.astype(np.float32)


STAGES = {0: reward_stage0, 1: reward_stage1, 2: reward_stage2, 3: reward_stage3, 4: reward_stage4, 5: reward_stage5, 6: reward_stage6}
