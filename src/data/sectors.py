"""
Sector and ownership classification for the Nifty 50 — the one place it is kept.

Codes are upper-case keys (they index the DCF's sector betas); LABELS gives the
display name. A symbol with no entry has sector "DEFAULT".
"""
from __future__ import annotations

SECTOR: dict[str, str] = {
    "HDFCBANK": "BANKING",  "ICICIBANK": "BANKING",  "SBIN": "BANKING",
    "AXISBANK": "BANKING",  "KOTAKBANK": "BANKING",  "INDUSINDBK": "BANKING",
    "TCS": "IT",  "INFY": "IT",  "HCLTECH": "IT",  "WIPRO": "IT",  "TECHM": "IT",
    "HINDUNILVR": "FMCG",  "ITC": "FMCG",  "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG",   "TATACONSUM": "FMCG",
    "TITAN": "FMCG",       "ASIANPAINT": "FMCG",
    "SUNPHARMA": "PHARMA", "DRREDDY": "PHARMA",  "CIPLA": "PHARMA",
    "MARUTI": "AUTO",  "TATAMOTORS": "AUTO",  "M&M": "AUTO",
    "BAJAJ-AUTO": "AUTO",  "EICHERMOT": "AUTO",  "HEROMOTOCO": "AUTO",
    "RELIANCE": "ENERGY",  "ONGC": "ENERGY",  "BPCL": "ENERGY",
    "NTPC": "ENERGY",      "POWERGRID": "ENERGY",  "COALINDIA": "ENERGY",
    "TATASTEEL": "METALS", "JSWSTEEL": "METALS",  "HINDALCO": "METALS",
    "LT": "INFRA",  "ADANIENT": "INFRA",  "ADANIPORTS": "INFRA",
    "BEL": "DEFENSE",
    "ULTRACEMCO": "CEMENT",  "GRASIM": "CEMENT",
    "BHARTIARTL": "TELECOM",
    "BAJFINANCE": "FINSERV",  "BAJAJFINSV": "FINSERV",  "SHRIRAMFIN": "FINSERV",
    "HDFCLIFE": "INSURANCE",  "SBILIFE": "INSURANCE",
    "APOLLOHOSP": "HEALTHCARE",
    "ZOMATO": "CONSUMER_TECH",
}

LABELS: dict[str, str] = {
    "BANKING": "Banking", "IT": "IT", "FMCG": "FMCG", "PHARMA": "Pharma",
    "AUTO": "Auto", "ENERGY": "Energy", "METALS": "Metals", "INFRA": "Infra",
    "DEFENSE": "Defence", "CEMENT": "Cement", "TELECOM": "Telecom",
    "FINSERV": "Fin-Serv", "INSURANCE": "Insurance", "HEALTHCARE": "Healthcare",
    "CONSUMER_TECH": "Consumer Tech", "DEFAULT": "Other",
}

# Government-controlled companies (the Government of India holds a majority).
PSU: frozenset[str] = frozenset({
    "SBIN", "ONGC", "BPCL", "NTPC", "POWERGRID", "COALINDIA", "BEL",
})


def sector_of(symbol: str) -> str:
    return SECTOR.get(symbol, "DEFAULT")


def sector_label(symbol: str) -> str:
    return LABELS[sector_of(symbol)]


def is_psu(symbol: str) -> bool:
    return symbol in PSU
