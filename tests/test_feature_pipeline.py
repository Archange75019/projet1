import unittest

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

from script.config import SECOND_TARGET, TARGET
from script.features import build_features
from script.modeling import CATEGORICAL_FEATURES, NUMERIC_FEATURES, make_model
from script.preprocessing import split_xy
from script.run_cleaning import clean_dataset


class FeaturePipelineTest(unittest.TestCase):
    def test_build_and_model_features_without_coordinates(self):
        emissions_target = SECOND_TARGET[0]
        clean = pd.DataFrame({
            "YearBuilt": [1990, 1960, 2000, 1980, 2010],
            "ListOfAllPropertyUseTypes": ["Office, Parking", "School", "Office", "Retail", "Office"],
            "LargestPropertyUseType": ["Office", "School", "Office", "Retail", "Office"],
            "LargestPropertyUseTypeGFA": [1000, 1200, 1500, 800, 900],
            "SecondLargestPropertyUseTypeGFA": [100, 0, 0, 200, 50],
            "ThirdLargestPropertyUseTypeGFA": [0, 0, 0, 10, 0],
            "PropertyGFATotal": [1200, 1200, 1500, 1100, 1000],
            "PropertyGFAParking": [100, 0, 0, 100, 50],
            "PropertyGFABuilding(s)": [1100, 1200, 1500, 1000, 950],
            "NumberofBuildings": [1, 1, 2, 0, 1],
            "NumberofFloors": [2, 3, 5, 1, 4],
            "BuildingType": ["NonResidential", "Campus", "NonResidential", "NonResidential", "Campus"],
            "PrimaryPropertyType": ["Office", "School", "Office", "Retail", "Office"],
            "SecondLargestPropertyUseType": ["Parking", "None", "None", "Office", "Parking"],
            "ThirdLargestPropertyUseType": ["None"] * 5,
            "Neighborhood": ["DOWNTOWN", "NORTH", "DOWNTOWN", "SOUTH", "NORTH"],
            "CouncilDistrictCode": [7, 5, 7, 2, 5],
            "ComplianceStatus": ["Compliant"] * 5,
            TARGET: [100000, 120000, 150000, 80000, 90000],
            emissions_target: [10, 20, 30, 40, 50],
        })

        raw = pd.concat([
            clean,
            clean.iloc[[0, 1]].assign(**{TARGET: [float("nan"), 0]}),
            clean.iloc[[0]].assign(ComplianceStatus="Error"),
        ], ignore_index=True)
        cleaned, _ = clean_dataset(raw, verbose=False)
        self.assertEqual(len(cleaned), len(clean))
        self.assertTrue(cleaned[TARGET].gt(0).all())

        featured = build_features(cleaned)
        self.assertNotIn("LogTarget", featured.columns)
        self.assertEqual(featured.loc[0, "HasMultipleUses"], 1)
        self.assertEqual(featured.loc[1, "HasMultipleUses"], 0)
        self.assertTrue(pd.isna(featured.loc[3, "GFAPerBuilding"]))
        X_train, X_test, y_train, y_test = split_xy(featured)

        self.assertNotIn("Latitude", X_train.columns)
        self.assertNotIn("Longitude", X_train.columns)
        self.assertNotIn(TARGET, X_train.columns)
        self.assertNotIn(emissions_target, X_train.columns)
        self.assertIn("GFAPerBuilding", NUMERIC_FEATURES)
        self.assertIn("LogGFAParking", NUMERIC_FEATURES)
        self.assertIn("HasMultipleUses", NUMERIC_FEATURES)
        self.assertIn("AgeClass", CATEGORICAL_FEATURES)
        self.assertIn("CouncilDistrictCode", CATEGORICAL_FEATURES)

        model = make_model(DummyRegressor(strategy="mean"))
        model.fit(X_train, y_train)
        self.assertEqual(len(model.predict(X_test)), len(y_test))

        fitted_pipeline = model.regressor_
        floors = fitted_pipeline.named_steps["floors"]
        preprocessor = fitted_pipeline.named_steps["prep"]
        X_test_unknown = X_test.copy()
        X_test_unknown["Neighborhood"] = "NOUVEAU_QUARTIER"
        train_encoded = preprocessor.transform(floors.transform(X_train))
        test_encoded = preprocessor.transform(floors.transform(X_test_unknown))
        self.assertEqual(train_encoded.shape[1], test_encoded.shape[1])
        self.assertEqual(len(preprocessor.get_feature_names_out()), test_encoded.shape[1])
        self.assertEqual(len(model.predict(X_test_unknown)), len(y_test))

        search = GridSearchCV(
            make_model(DummyRegressor()),
            [
                {
                    "regressor__model": [DummyRegressor()],
                    "regressor__model__strategy": ["mean", "median"],
                },
                {
                    "regressor__model": [LinearRegression()],
                    "regressor__model__fit_intercept": [True, False],
                },
            ],
            cv=2,
            scoring="neg_root_mean_squared_error",
            error_score="raise",
        )
        X_grid = featured[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        search.fit(X_grid, featured[TARGET])
        self.assertEqual(len(search.predict(X_grid)), len(featured))

if __name__ == "__main__":
    unittest.main()