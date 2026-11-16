"""Constantes partagées par tous les scripts de nettoyage."""

TARGET = "SiteEnergyUse(kBtu)"  # cible choisie pour tout le projet

# --- Règles de filtrage des lignes -------------------------------------------
MAX_FLOORS = 80          # la plus haute tour de Seattle a 76 étages
YEAR_MIN, YEAR_MAX = 1800, 2016
KEEP_COMPLIANCE = "Compliant"
TUKEY_COEF = 1.5         # coefficient IQR pour les outliers de la cible (en log10)

# Outliers d'intensité énergétique (cible / surface), par type d'usage principal.
# Une ligne est écartée si son log10(intensité) est à plus de K écarts interquartiles
# de la médiane de son type. Les types de moins de EUI_MIN_GROUP bâtiments sont
# comparés à l'ensemble du jeu. L'intensité sert UNIQUEMENT à détecter, jamais de feature.
EUI_IQR_K = 3.0
EUI_MIN_GROUP = 10

# Surfaces par usage : somme des 3 usages > GFA_SUM_TOL x surface totale = incohérent
GFA_SUM_TOL = 1.05

# --- Règles sur les colonnes -------------------------------------------------
# NaN signifiant "aucun" (bâtiment à usage unique) et non "inconnu"
FILL_NONE_COLS = ["SecondLargestPropertyUseType", "ThirdLargestPropertyUseType"]
FILL_ZERO_COLS = ["SecondLargestPropertyUseTypeGFA", "ThirdLargestPropertyUseTypeGFA"]

# Identifiants sans valeur prédictive
ID_COLS = ["PropertyName", "Address", "TaxParcelIdentificationNumber", "ZipCode"]

# Colonnes calculées à partir de la consommation : DATA LEAKAGE.
LEAKAGE_COLS = [
    "SiteEUI(kBtu/sf)", "SiteEUIWN(kBtu/sf)",
    "SourceEUI(kBtu/sf)", "SourceEUIWN(kBtu/sf)",
    "SiteEnergyUseWN(kBtu)",
    "SteamUse(kBtu)", "Electricity(kWh)", "Electricity(kBtu)",
    "NaturalGas(therms)", "NaturalGas(kBtu)",
    "TotalGHGEmissions", "GHGEmissionsIntensity",
    "ENERGYSTARScore", "YearsENERGYSTARCertified",
]
# Gardée dans le fichier "modèle" pour l'extension double cible (c'est une 2e cible, pas une feature)
SECOND_TARGET = ["TotalGHGEmissions"]

NEIGHBORHOOD_ALIASES = {"DELRIDGE NEIGHBORHOODS": "DELRIDGE"}

# --- Sorties -----------------------------------------------------------------
CLEAN_FILENAME = "2016-building-energy-benchmarking-clean.csv"
MODEL_FILENAME = "2016-building-energy-benchmarking-model.csv"

# --- Feature engineering -----------------------------------------------------
REF_YEAR = 2016
SEATTLE_CENTER = (47.6062, -122.3321)  # (latitude, longitude) approximatifs
AGE_BINS = [-1, 20, 50, 80, 200]
AGE_LABELS = ["<=20 ans", "21-50 ans", "51-80 ans", ">80 ans"]
HIGH_RISE_FLOORS = 10

# Familles d'usage : règles ordonnées (première règle qui correspond gagne).
# Regroupement métier FIXE (pas basé sur des fréquences) : aucune fuite train/test.
USE_FAMILY_RULES = [
    ("Parking", ["parking"]),
    ("Lab / Data center", ["laboratory", "data center"]),
    ("Health", ["hospital", "medical", "clinic", "urgent care", "outpatient"]),
    ("Office", ["office", "bank", "financial"]),
    ("Warehouse / Storage", ["warehouse", "distribution", "storage"]),
    ("Retail / Food", ["retail", "supermarket", "grocery", "mall", "dealership",
                       "lifestyle", "restaurant", "bar"]),
    ("Lodging / Care", ["hotel", "residence hall", "dormitory", "lodging",
                        "senior care", "residential care"]),
    ("Education", ["k-12", "school", "college", "university", "pre-school",
                   "daycare", "education"]),
    ("Assembly / Leisure", ["worship", "meeting", "entertainment", "recreation",
                            "museum", "performing", "movie", "fitness", "library"]),
    ("Industrial", ["manufacturing", "industrial", "utility", "repair"]),
    ("Public services", ["police", "fire station", "courthouse", "prison",
                         "personal services", "public services", "services"]),
]

# Colonnes jamais données au modèle
# - identifiants / texte libre à haute cardinalité
# - LargestPropertyUseType (52 valeurs) : remplacée par LargestUseFamily
NON_FEATURE_COLS = ["OSEBuildingID", "ListOfAllPropertyUseTypes", "LargestPropertyUseType"]

# Ne pas utiliser OneHotEncoder sans limite sur les variables à haute cardinalité
OHE_MAX_CATEGORIES = 15