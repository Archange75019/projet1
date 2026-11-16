"""Journal du nettoyage : lignes supprimées (en nombre et en %) à chaque étape,
et lignes modifiées / signalées (sans suppression) par certaines règles."""
import pandas as pd


def _pct(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole else 0.0


class CleaningLog:
    def __init__(self, verbose: bool = True):
        self.steps: list[dict] = []
        self.touched: list[dict] = []
        self.verbose = verbose
        self.n_initial: int | None = None   # taille du jeu brut (référence du % cumulé)

    def step(self, label: str, before: int, after: int) -> None:
        if self.n_initial is None:
            self.n_initial = before
        lost = before - after
        self.steps.append({
            "étape": label,
            "avant": before,
            "après": after,
            "supprimées": lost,
            "% perdu vs étape précédente": round(_pct(lost, before), 2),
            "% perdu vs brut": round(_pct(lost, self.n_initial), 2),
            "% restant vs brut": round(_pct(after, self.n_initial), 2),
        })
        if self.verbose:
            print(f"{label} : {before} -> {after} "
                  f"(-{lost} lignes, -{_pct(lost, before):.1f}% vs précédent, "
                  f"{_pct(after, self.n_initial):.1f}% du brut restant)")

    def touch(self, label: str, n_rows: int, total: int | None = None) -> None:
        """Règle qui MODIFIE ou SIGNALE des lignes sans les supprimer.
        `total` = taille du jeu à ce moment, pour exprimer la part concernée en %."""
        row = {"règle": label, "lignes concernées": int(n_rows)}
        if total:
            row["% du jeu"] = round(_pct(n_rows, total), 2)
        self.touched.append(row)
        if self.verbose:
            suffix = f" ({_pct(n_rows, total):.1f}% du jeu)" if total else ""
            print(f"{label} : {int(n_rows)} lignes concernées{suffix}, conservées")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    def touched_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.touched)

    def summary(self) -> str:
        """Bilan global : combien du jeu brut a été conservé / perdu."""
        if not self.steps:
            return "Aucune étape enregistrée."
        final = self.steps[-1]["après"]
        lost = self.n_initial - final
        return (f"Bilan : {self.n_initial} -> {final} lignes "
                f"(-{lost}, soit {_pct(lost, self.n_initial):.1f}% de perte au total, "
                f"{_pct(final, self.n_initial):.1f}% conservé)")