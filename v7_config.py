# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — CONFIG
Kurumsal Algo Trading Bot · Tüm ayarlar burada
"""
from __future__ import annotations

import os
from typing import Dict, List

# ═══════════════════════════════════════════════════════════════════
# 1. SÜRÜM VE KİMLİK
# ═══════════════════════════════════════════════════════════════════
BOT_VERSION = "V7-APEX-1.0"
BOT_NAME = "🐋 V7 APEX"
BOT_SLOGAN = "Zamanı oku, sabrı öğren, kazan."

# ═══════════════════════════════════════════════════════════════════
# 2. SEKTÖRLER VE HİSSELER (3 Bot × 24 Hisse)
# ═══════════════════════════════════════════════════════════════════
SEKTORLER: Dict[str, Dict] = {
    "BANKACILIK": {
        "bot_id": 1,
        "sermaye": 33333.0,
        "hisseler": [
            "GARAN", "AKBNK", "ISCTR", "YKBNK", "HALKB", "VAKBN",
            "QNBFB", "TSKB", "ALBRK", "SKBNK", "ICBCT", "KLNMA",
            "QNBFL", "KCHOL", "SAHOL", "AGHOL", "GLYHO", "TURSG",
            "ANSGR", "AKGRT", "ISMEN", "ISFIN", "HLGYO", "ISGYO",
        ],
        "aciklama": "Bankacılık + Holding + Sigorta + GYO",
        "risk_profil": {
            "risk_per_trade": 0.008,
            "max_positions": 6,
            "sl_atr": 2.0,
            "daily_loss": 2.0,
        },
    },
    "HAVACILIK": {
        "bot_id": 2,
        "sermaye": 33334.0,
        "hisseler": [
            "THYAO", "PGSUS", "TAVHL", "CLEBI", "DOAS", "FROTO",
            "TOASO", "TTRAK", "OTKAR", "ASUZU", "KARSN", "BRISA",
            "GOODY", "TUKAS", "ULKER", "CCOLA", "AEFES", "BIMAS",
            "MGROS", "SOKM", "MAVI", "MEPET", "BIZIM", "ULUFA",
        ],
        "aciklama": "Havacılık + Otomotiv + Perakende + Gıda",
        "risk_profil": {
            "risk_per_trade": 0.012,
            "max_positions": 8,
            "sl_atr": 3.0,
            "daily_loss": 3.5,
        },
    },
    "ENERJI": {
        "bot_id": 3,
        "sermaye": 33333.0,
        "hisseler": [
            "TUPRS", "PETKM", "AKSEN", "ENJSA", "ENERY", "ZOREN",
            "ODAS", "AYDEM", "EREGL", "KRDMD", "ISDMR", "SISE",
            "TRKCM", "SASA", "AKSA", "GUBRF", "ASELS", "LOGO",
            "NETAS", "ARDYZ", "KAREL", "KONTR", "PAPIL", "FORTE",
        ],
        "aciklama": "Enerji + Demir-Çelik + Kimya + Teknoloji",
        "risk_profil": {
            "risk_per_trade": 0.010,
            "max_positions": 7,
            "sl_atr": 2.5,
            "daily_loss": 3.0,
        },
    },
}

SEKTOR_ENV = os.environ.get("V7_SEKTOR", "BANKACILIK").upper()
AKTIF_SEKTOR = SEKTOR_ENV if SEKTOR_ENV in SEKTORLER else "BANKACILIK"

BOT_ID = SEKTORLER[AKTIF_SEKTOR]["bot_id"]
BOT_SERMAYE = SEKTORLER[AKTIF_SEKTOR]["sermaye"]
STOCK_LIST: List[str] = SEKTORLER[AKTIF_SEKTOR]["hisseler"]
N_STOCKS = len(STOCK_LIST)

# ═══════════════════════════════════════════════════════════════════
# 3. GENEL SABİTLER
# ═══════════════════════════════════════════════════════════════════
HORIZON = 10
HORIZON_LONG = 40
TB_K = 1.2
TB_K_LONG = 1.5
ALPHA_LONG = 0.35

FEE = 0.0015
SLIPPAGE = 0.001
RT_COST = 2 * (FEE + SLIPPAGE)

HW = 90
WARMUP_BARS = 60
SEQ_LEN = 30
TIME_DECAY_HALF_LIFE = 504
CORR_WINDOW = 60
PEER_K = 3
LOG_ADV_NORM = 22.03

# ═══════════════════════════════════════════════════════════════════
# 4. ÖZELLİK İSİMLERİ (306 Özellik)
# ═══════════════════════════════════════════════════════════════════
TECH_NAMES: List[str] = [
    "rsi7", "rsi14", "rsi21",
    "atrn7", "atrn14", "atrn21",
    "macd", "macd_sig", "macd_hist",
    "bb_pct", "bb_width",
    "stoch_k", "stoch_d",
    "vwap_dist", "obv_norm",
    "ret1", "ret2", "ret3", "ret5", "ret10", "ret20", "ret40",
    "sma5", "sma10", "sma20", "sma50", "sma200",
    "ema12", "ema26",
    "rv10", "rv30", "rv60",
    "gk20", "pk20",
    "hl", "co", "wick_up", "wick_lo",
    "vol_rel", "vol_chg",
    "amihud", "cs_spread",
    "gap",
    "skew20", "kurt20",
    "dd60", "dist_low60",
    "mom_acc", "eff_ratio20",
]

TECH_NAMES_EXT: List[str] = [
    "tenkan", "kijun", "senkou_a", "senkou_b", "chikou",
    "keltner_up", "keltner_lo", "keltner_width",
    "donchian_up", "donchian_lo", "donchian_width",
    "aroon_up", "aroon_dn", "aroon_osc",
    "cci", "roc5", "roc10", "roc20",
    "willr7", "willr14", "willr21",
    "adx", "plus_di", "minus_di",
    "psar", "psar_trend",
    "vortex_pos", "vortex_neg",
    "trix", "dpo", "mass_idx",
    "chaikin_osc", "force_idx", "eom",
    "klinger", "elder_bull", "elder_bear",
    "ultimate_osc", "awesome_osc", "bop",
    "coppock", "fisher", "schaff",
    "kst", "ppo", "ppo_sig", "ppo_hist",
    "rvi", "stochrsi_k", "stochrsi_d",
    "connors_rsi", "qstick",
    "vwma", "hull_ma", "alma",
    "tema", "dema", "zlema",
    "mcginley", "kama", "frama", "mama",
]

TECH_ALL_NAMES = TECH_NAMES + TECH_NAMES_EXT
N_TECH = len(TECH_ALL_NAMES)

LIQ_NAMES: List[str] = [
    "adv_log", "turnover_ratio", "depth_bid", "depth_ask",
    "resilience", "tightness", "elasticity", "kyle_lambda",
    "amihud_illiq", "roll_spread", "cs_spread_ext",
    "hasbrouck", "vpin", "volume_sync", "tick_size_eff",
]

RANK_NAMES: List[str] = [
    "rk_ret5", "rk_ret20", "rk_ret60", "rk_vol", "rk_volrel",
    "rk_rsi", "rk_macd", "rk_bb", "rk_amihud", "rk_turnover",
    "rk_sector_ret5", "rk_sector_ret20", "rk_peer_ret5", "rk_peer_ret20",
    "rk_global_ret5", "rk_global_ret20", "rk_size", "rk_mom",
    "rk_value", "rk_quality",
]

MACRO_KEYS = ["usdtry", "eurtry", "gold", "brent", "vix", "dxy", "oil", "copper"]
MACRO_NAMES: List[str] = (
    [f"M_{k}_{s}" for k in MACRO_KEYS for s in ("z", "r20", "r60")]
    + ["M_yield_curve", "M_credit_spread", "M_fed_rate",
       "M_inflation", "M_unemployment", "M_gdp"]
)

MICRO_NAMES: List[str] = [
    "ofi", "ofi_ema5", "ofi_ema20", "ofi_l1", "ofi_l2", "ofi_l3",
    "queue_bid", "queue_ask", "queue_mid", "iceberg_bid", "iceberg_ask",
    "iceberg_conf", "spoof_bid", "spoof_ask", "spoof_sev",
    "micro_price", "weighted_mid", "mid_price",
    "spread_eff", "spread_quoted", "spread_real",
    "depth_proxy", "kyle_lambda_ext",
    "hawkes_lambda", "hawkes_bid", "hawkes_ask",
    "trade_imb", "vol_imb", "order_imb",
    "tick_rule", "lee_ready", "trade_sign",
    "lob_slope_bid", "lob_slope_ask", "lob_curv_bid", "lob_curv_ask",
    "lob_asym", "hhi_bid", "hhi_ask", "hhi_comb",
]
N_MICRO = len(MICRO_NAMES)

PEER_NAMES: List[str] = [
    "peer_ret_w", "rel_strength", "peer_corr_mean", "peer_mom",
    "sector_ret", "beta_sector", "idio_ret", "lead_lag",
    "peer_ret5", "peer_ret20", "peer_vol", "peer_rsi", "peer_macd",
    "sector_ret5", "sector_ret20", "sector_vol", "sector_breadth",
    "multi_hop_1", "multi_hop_2", "multi_hop_3",
    "dynamic_corr_5", "dynamic_corr_20", "dynamic_corr_60",
    "cross_asset_spill", "tail_dep",
]

NLP_NAMES: List[str] = [
    "news_sent_1d", "news_sent_5d", "news_sent_20d",
    "news_count_1d", "news_count_5d",
    "news_conf_mean", "news_conf_std",
    "kap_count_1d", "kap_count_7d", "kap_impact_score",
    "social_sent", "social_volume", "social_divergence",
    "finbert_pos", "finbert_neg",
]

SECTORS = ["BANKACILIK", "HAVACILIK", "ENERJI", "DIGER"]
REGIMES = ["BULL", "BEAR", "VOL", "RANGE"]
STOCK_ONEHOT_NAMES = [f"stk_{c}" for c in STOCK_LIST]
SECTOR_ONEHOT_NAMES = [f"sec_{s}" for s in SECTORS]
REGIME_ONEHOT_NAMES = [f"reg_{r}" for r in REGIMES]
CAP_ONEHOT_NAMES = [f"cap_{c}" for c in ("small", "mid", "large")]
TIME_ONEHOT_NAMES = [f"time_{t}" for t in ("open", "mid", "close")]
OTHER_ONEHOT_NAMES = ["oh_month_end", "oh_quarter_end"]
ONEHOT_NAMES = (
    STOCK_ONEHOT_NAMES + SECTOR_ONEHOT_NAMES + REGIME_ONEHOT_NAMES
    + CAP_ONEHOT_NAMES + TIME_ONEHOT_NAMES + OTHER_ONEHOT_NAMES
)

CAUSAL_NAMES: List[str] = [
    "causal_parent", "causal_child", "causal_path",
    "do_effect", "counterfactual", "mediation",
    "instrument", "confounder", "collider", "d_separation",
]

FEATURE_NAMES: List[str] = (
    TECH_ALL_NAMES + LIQ_NAMES + RANK_NAMES + MACRO_NAMES + MICRO_NAMES
    + PEER_NAMES + NLP_NAMES + ONEHOT_NAMES + CAUSAL_NAMES
)
N_FEAT = len(FEATURE_NAMES)

# ═══════════════════════════════════════════════════════════════════
# 5. MODEL HİPER-PARAMETRELERİ
# ═══════════════════════════════════════════════════════════════════
H_DIM = 64
BAG_N = 5
BAG_SEED = 100
HOLD_N = 6000

GAT_HEADS = 4
GAT_DIM = 32
GAT_LAYERS = 2

TFT_DIM = 64
TFT_HEADS = 4
TFT_LAYERS = 3

INFORMER_SEQ = 500
INFORMER_HEADS = 4
INFORMER_LAYERS = 2

N_EXPERTS = 16
MOE_TOPK = 4
LB_COEF = 0.01
ZLOSS_COEF = 1e-3

CHRONOS_MODEL = "amazon/chronos-t5-small"
CHRONOS_PRED_LEN = 40

MC_SAMPLES = 20
USE_ROUTER = True
USE_DSR = True

# ═══════════════════════════════════════════════════════════════════
# 6. DOĞRULAMA (CV)
# ═══════════════════════════════════════════════════════════════════
CPCV_ENABLED = True
CPCV_N_GROUPS = 8
CPCV_N_TEST_GROUPS = 4
PURGE_BARS = HORIZON_LONG
EMBARGO_BARS = 5
PBO_ENABLED = True

# ═══════════════════════════════════════════════════════════════════
# 7. MARKET IMPACT
# ═══════════════════════════════════════════════════════════════════
DYNAMIC_SLIPPAGE = True
SLIPPAGE_BASE = 0.0005
AC_ETA = 0.142
AC_GAMMA = 0.314
AC_BETA = 0.6

HAWKES_MU = 0.1
HAWKES_ALPHA = 0.5
HAWKES_BETA = 1.0

OFI_WINDOW = 20

# ═══════════════════════════════════════════════════════════════════
# 8. RİSK PARAMETRELERİ
# ═══════════════════════════════════════════════════════════════════
RISK_PARITY = True
VAR_CONFIDENCE = 0.95
CVAR_ALPHA = 0.95

DEFAULT_RISK: Dict[str, float] = {
    "risk_per_trade": 0.01,
    "max_pos": 0.15,
    "max_positions": 8,
    "max_exposure": 0.85,
    "max_sector": 0.40,
    "sl_atr": 2.5,
    "trail_act_r": 1.5,
    "trail_atr": 2.5,
    "tp_on": True,
    "tp_r": 2.0,
    "tp_frac": 0.5,
    "be_on": True,
    "be_r": 1.0,
    "max_hold": 120,
    "p_buy": 0.45,
    "p_sell": 0.45,
    "p_buy_long": 0.40,
    "margin": 0.10,
    "unc_max": 0.25,
    "conf_sizing": True,
    "max_dd": 15.0,
    "daily_loss": 3.0,
    "max_trades_day": 5,
    "loss_streak": 3,
    "cooldown_h": 24,
    "dd_throttle": True,
    "use_regime": True,
    "use_long_gate": True,
    "regime_bull_adj": -0.03,
    "regime_bear_adj": 0.08,
    "regime_range_adj": 0.03,
    "regime_vol_adj": 0.10,
    "kelly_frac": 0.25,
    "kelly_unc_pen": 2.0,
    "cvar_alpha": 0.95,
    "cvar_limit": 0.03,
    "max_adv_pct": 0.05,
    "impact_on": True,
    "ofi_gate": True,
    "ofi_min": -0.30,
    "corr_on": True,
    "corr_max": 0.85,
    "si_auto": True,
    "si_interval_min": 30,
    "si_budget_s": 30,
    "dsr_confidence": 0.90,
    "consec_improve": 3,
    "auto_adopt": True,
    "min_daily_turnover": 5_000_000,
}

DEFAULT_RISK.update(SEKTORLER[AKTIF_SEKTOR]["risk_profil"])

# ═══════════════════════════════════════════════════════════════════
# 9. API VE DOSYA YOLLARI
# ═══════════════════════════════════════════════════════════════════
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

DB_FILE = f"v7_{AKTIF_SEKTOR.lower()}.duckdb"
STATE_FILE = f"v7_{AKTIF_SEKTOR.lower()}.pkl"
MODEL_FILE = f"v7_{AKTIF_SEKTOR.lower()}.pt"
CACHE_DIR = "cache_v7"
os.makedirs(CACHE_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# 10. YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════════
def get_risk_params(sektor: str = AKTIF_SEKTOR) -> Dict:
    """Sektör risk profilini varsayılanlarla birleştirir."""
    rp = dict(DEFAULT_RISK)
    if sektor in SEKTORLER:
        rp.update(SEKTORLER[sektor]["risk_profil"])
    return rp


def feature_index(name: str) -> int:
    """Özellik adından indeks döndürür."""
    return FEATURE_NAMES.index(name)


if __name__ == "__main__":
    print(f"🐋 {BOT_NAME} {BOT_VERSION}")
    print(f"Sektör: {AKTIF_SEKTOR}")
    print(f"Hisse sayısı: {N_STOCKS}")
    print(f"Toplam özellik: {N_FEAT}")
    print(f"  - Teknik: {N_TECH}")
    print(f"  - Likidite: {len(LIQ_NAMES)}")
    print(f"  - Rank: {len(RANK_NAMES)}")
    print(f"  - Makro: {len(MACRO_NAMES)}")
    print(f"  - Mikro: {N_MICRO}")
    print(f"  - Peer: {len(PEER_NAMES)}")
    print(f"  - NLP: {len(NLP_NAMES)}")
    print(f"  - One-hot: {len(ONEHOT_NAMES)}")
    print(f"  - Causal: {len(CAUSAL_NAMES)}")
    print(f"Risk parametreleri: {len(DEFAULT_RISK)} adet")
