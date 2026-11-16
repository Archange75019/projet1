"""Journal du nombre de lignes avant / après chaque étape de nettoyage,
et des lignes modifiées (sans suppression) par certaines règles."""
import pandas as pd


class CleaningLog:
    def __init__(self, verbose: bool = True):
        self.steps: list[dict] = []
        self.touched: list[dict] = []
        self.verbose = verbose

    def step(self, label: str, before: int, after: int) -> None:
        self.steps.append(
            {"étape": label, "avant": before, "après": after, "supprimées": before - after}
        )
        if self.verbose:
            print(f"{label} : {before} -> {after} ({before - after} lignes supprimées)")

    def touch(self, label: str, n_rows: int) -> None:
        """Règle qui MODIFIE ou SIGNALE des lignes sans les supprimer."""
        self.touched.append({"règle": label, "lignes concernées": int(n_rows)})
        if self.verbose:
            print(f"{label} : {int(n_rows)} lignes concernées (conservées)")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    def touched_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.touched)