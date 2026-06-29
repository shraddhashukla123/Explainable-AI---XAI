from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import shap
import matplotlib.pyplot as plt

from xai_runner.preprocess.type_detection import (
    detect_datatypes,
    sanitize_numeric_strings,
    parse_bracketed_numeric_series,
    build_column_summary,
)
from xai_runner.utils.loggers import get_logger
from xai_runner.config import OUTPUT_DIR
from xai_runner.driver_analysis import (
    run_bivariate_regression,
    run_bivariate_logistic,
)

logger: logging.Logger = get_logger("XAI_Driver_Engine")


ECONOMIC_PREFIX_MAP = {
    "GDP_VALUE": "Economic_GDP",
    "UNEMPLOYMENT_RATE": "Economic_Unemployment",
    "CPIH_VALUE": "Economic_CPIH",
    "RPI_VALUE": "Economic_RPI",
}

WEATHER_PREFIX_MAP = {
    "Temperature": "Weather_Temperature",
    "Minimum Temperature": "Weather_Minimum Temperature",
    "Maximum Temperature": "Weather_Maximum Temperature",
    "Precipitation": "Weather_Precipitation",
    "Total Snowfall": "Weather_Total Snowfall",
    "Wind Speed": "Weather_Wind Speed",
    "Wind Gust": "Weather_Wind Gust",
    "Atmospheric Pressure": "Weather_Atmospheric Pressure",
}

def apply_driver_prefixes(X: pd.DataFrame) -> pd.DataFrame:
    if X is None or X.empty:
        return X

    rename_map = {}

    for c in X.columns:
        base = c.rsplit("_", 1)[0] if "_" in c and c.rsplit("_", 1)[-1].isdigit() else c

        if base in ECONOMIC_PREFIX_MAP:
            rename_map[c] = c.replace(base, ECONOMIC_PREFIX_MAP[base], 1)

        elif base in WEATHER_PREFIX_MAP:
            rename_map[c] = c.replace(base, WEATHER_PREFIX_MAP[base], 1)

    return X.rename(columns=rename_map) if rename_map else X


def _build_shap_aggregate_table(
    shap_values: np.ndarray,
    feature_names: List[str],
    model_name: str,
    transformation: str,
    target: str,
) -> pd.DataFrame:

    vals = np.asarray(shap_values)

    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)

    n_samples = vals.shape[0]

    pct_positive = (vals > 0).sum(axis=0) / n_samples * 100.0
    pct_negative = (vals < 0).sum(axis=0) / n_samples * 100.0
    abs_mean = np.abs(vals).mean(axis=0)
    overall_mean = vals.mean(axis=0)

    # agg_df = pd.DataFrame(
    #     {

    #         "model_name": model_name,
    #         "transformation": transformation,
    #         "target": target,
    #         "drivers": feature_names,
    #         "pct_positive": pct_positive,
    #         "pct_negative": pct_negative,
    #         "absolute_mean": abs_mean,
    #         "overall_mean": overall_mean,
    #     }
    # )

    encoded = pd.Series(feature_names, name="encoded_drivers")

    def map_original_driver(x: str) -> str:
        if x.startswith("Economic_"):
            return "Economic"
        elif x.startswith("Weather_"):
            return "Weather"
        elif x.startswith("EVENT_"):
            return "EVENT"
        elif x.startswith("Holiday_Name_"):
            return "Holiday"
        elif x.startswith("Qtr_"):
            return "Qtr"
        elif x.startswith("Period_"):
            return "Period"
        else:
            return "Other"

    original = encoded.apply(map_original_driver)

    agg_df = pd.DataFrame(
        {
            "model_name": model_name,
            "transformation": transformation,
            "target": target,
            "original_drivers": original.values,
            "encoded_drivers": encoded.values,
            "pct_positive": pct_positive,
            "pct_negative": pct_negative,
            "absolute_mean": abs_mean,
            "overall_mean": overall_mean,
        }
    )



 
    agg_df = agg_df.sort_values("absolute_mean", ascending=False).reset_index(drop=True)
    return agg_df



class ExplainableModelRunner:
    def __init__(self, dataframe: pd.DataFrame):
        self.df_raw: Optional[pd.DataFrame] = dataframe.copy() if dataframe is not None else None
        self.df: Optional[pd.DataFrame] = dataframe.copy() if dataframe is not None else None

        # Target info
        self.target_col: Optional[str] = None
        self.target_series_raw: Optional[pd.Series] = None
        self.y_raw: Optional[pd.Series] = None
        self.y: Optional[pd.Series] = None
        self.problem_type: Optional[str] = None
        self.class_mapping: Dict[Any, int] = {}

        # Feature buckets
        self.nominal_features: List[str] = []
        self.numeric_features: List[str] = []
        self.identifier_features: List[str] = []
        self.high_cardinality_nominal: List[str] = []
        self.schema_full: Optional[pd.DataFrame] = None

        # Storage for driver outputs
        self.driver_results_raw: Optional[pd.DataFrame] = None
        self.driver_results_minmax: Optional[pd.DataFrame] = None
        self.shap_agg_minmax: Optional[pd.DataFrame] = None
        self.shap_values_minmax: Optional[pd.DataFrame] = None

        self.driver_results_logistic: Optional[pd.DataFrame] = None

        # Model matrices
        self.preprocessed_df: Optional[pd.DataFrame] = None
        self.X_base: Optional[pd.DataFrame] = None
        self.X_raw: Optional[pd.DataFrame] = None
        self.X_minmax: Optional[pd.DataFrame] = None

        # Output dirs
        # self.run_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        # self.output_dir: Path = Path(OUTPUT_DIR) / self.run_id
        self.run_id: str = "latest"
        self.output_dir: Path = Path(OUTPUT_DIR)

        self.output_dir = Path(OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)


        self.logger = get_logger("ExplainableModelRunner")
        self.logger.info(f"ExplainableModelRunner initialised | run_id={self.run_id}")

    def _save_table(self, df: pd.DataFrame, save_path: Path | str) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if df.columns.duplicated().any():
            new_cols = []
            seen: Dict[str, int] = {}
            for c in df.columns:
                if c not in seen:
                    seen[c] = 0
                    new_cols.append(c)
                else:
                    seen[c] += 1
                    new_cols.append(f"{c}__dup{seen[c]}")
            df = df.copy()
            df.columns = new_cols

        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"Saved CSV: {save_path}")

    def select_target_column(self) -> None:
        if self.df_raw is None:
            raise ValueError("Dataframe not loaded in runner.")

        print("\nSelect Target Column (Y)\n")
        for idx, col in enumerate(self.df_raw.columns, start=1):
            print(f"  {idx}. {col}")

        while True:
            user_in = input("\nEnter target column (Y): ").strip()
            if user_in in self.df_raw.columns:
                self.target_col = user_in
                break
            else:
                print("Invalid column name. Try again.\n")

        self.logger.info(f"Target column selected: {self.target_col}")
        print(f"\nTarget Column (Y): {self.target_col}\n")

        self.target_series_raw = self.df_raw[self.target_col]

        print("Raw Data:\n")
        print(self.df_raw.head(), "\n")

    def detect_feature_types(self) -> None:
        if self.df_raw is None:
            raise ValueError("Dataframe not set.")
        if self.target_col is None:
            raise ValueError("Target column not selected.")

        print("\nFeature Type Detection\n")

        df_for_schema = self.df_raw.copy()

        # Period: YYYYMM → month (1–12)
        if "Period" in df_for_schema.columns:
            df_for_schema["Period"] = (
                df_for_schema["Period"]
                .astype(str)
                .str[-2:]
                .astype(int)
            )

        # Qtr: YYYYQ → quarter (1–4)
        if "Qtr" in df_for_schema.columns:
            df_for_schema["Qtr"] = (
                df_for_schema["Qtr"]
                .astype(str)
                .str[-1]
                .astype(int)
            )

        #schema_full = detect_datatypes(self.df_raw)
        schema_full = detect_datatypes(df_for_schema)
        self.schema_full = schema_full
        print("Column Nature Summary (internal schema):\n")
        print(schema_full, "\n")


        #column_summary = build_column_summary(self.df_raw, schema_full)
        column_summary = build_column_summary(df_for_schema, schema_full)
        summary_path = self.output_dir / "column_summary.csv"
        self._save_table(column_summary, summary_path)
        print(f"Column summary saved to: {summary_path}\n")

        self.identifier_features = schema_full.loc[
            schema_full["is_identifier"], "column"
        ].tolist()

        self.high_cardinality_nominal = schema_full.loc[
            schema_full["is_high_cardinality"], "column"
        ].tolist()

        schema_mod = schema_full[
            (schema_full["use_for_model"]) & (schema_full["column"] != self.target_col)
        ]

        self.nominal_features = schema_mod.loc[
            (schema_mod["likely_categorical"]) & (~schema_mod["is_high_cardinality"]),
            "column",
        ].tolist()

     
        self.numeric_features = schema_mod.loc[
            ~(schema_mod["likely_categorical"]),
            "column",
        ].tolist()

        if self.target_col == "Sales_TY" and "Sales_LY" in self.numeric_features:
            self.numeric_features.remove("Sales_LY")

        print(f"Identifier Columns: {self.identifier_features}\n")
        print(f"High-cardinality Nominals (dropped from model): {self.high_cardinality_nominal}\n")
        print(
            f"Nominal Columns used for model ({len(self.nominal_features)}): "
            f"{self.nominal_features}\n"
        )
        print(
            f"Numeric Columns used for model ({len(self.numeric_features)}): "
            f"{self.numeric_features}\n"
        )

        logger.info(
            "Type detection complete | nominal=%d | numeric=%d",
            len(self.nominal_features),
            len(self.numeric_features),
        )


    def preprocess_data(self) -> None:
        if self.df_raw is None or self.target_col is None:
            raise ValueError("Dataframe or target column not set.")

        print("\nData Preprocessing\n")

        df = self.df_raw.copy()
        print("Original Data:\n")
        print(df.head(), "\n")

        
        if "Period" in df.columns:
            df["Period"] = (
                df["Period"]
                .astype(str)
                .str[-2:]
                .astype(int)
            )

        
        if "Qtr" in df.columns:
            df["Qtr"] = (
                df["Qtr"]
                .astype(str)
                .str[-1]
                .astype(int)
            )

       
        drop_calendar_cols = [
            "Year",
            "Yr_week",
            "Holiday_Flag",
        ]

        df = df.drop(columns=[c for c in drop_calendar_cols if c in df.columns], errors="ignore")

        # Drop identifier columns
        if self.identifier_features:
            df = df.drop(columns=self.identifier_features, errors="ignore")
            print(f"Dropped Identifiers: {self.identifier_features}\n")

        
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[self.target_col])

        y = df[self.target_col]
        X = df.drop(columns=[self.target_col])

        
        keep_cols = sorted(
            set(self.numeric_features).union(self.nominal_features).intersection(X.columns)
        )
        X = X[keep_cols]

        
        for col in list(self.numeric_features):
            if col not in X.columns:
                continue
            s = X[col]
            conv = parse_bracketed_numeric_series(s)
            med = conv.median()
            X[col] = conv.fillna(med)

        
        for col in list(self.nominal_features):
            if col not in X.columns:
                continue
            # X[col] = (
            #     X[col]
            #     .astype(str)
            #     .replace({"Unknown": "Missing", "": "Missing"})
            #     .fillna("Missing")
            # )
            X[col] = (
                X[col]
                .fillna("Missing")                      # handle real NaN first
                .replace({"Unknown": "Missing", "": "Missing", "nan": "Missing"})
                .astype(str)
            )

        print("After Numeric & Categorical Cleaning:\n")
        print(X.head(), "\n")

        self.preprocessed_df = pd.concat([X, y], axis=1).reset_index(drop=True)
        self.X_base = X.reset_index(drop=True)
        self.y_raw = y.reset_index(drop=True)

        cleaned_path = self.output_dir / "cleaned_raw_data.csv"
        self._save_table(self.preprocessed_df, cleaned_path)
        print(f"Cleaned raw data (with target) saved to: {cleaned_path}\n")

        logger.info(
            "Preprocessing complete | shape=%s",
            self.preprocessed_df.shape,
        )


    # def confirm_problem_type(self) -> str:
    #     if self.y_raw is None:
    #         raise ValueError("Run preprocess_data before confirm_problem_type.")

    #     y = self.y_raw
    #     n_unique = y.nunique(dropna=True)

    #     # Auto-detection logic
    #     if pd.api.types.is_numeric_dtype(y):
    #         if n_unique == 2:
    #             auto_type = "classification"
    #         else:
    #             auto_type = "regression"
    #     else:
    #         if n_unique <= 10:
    #             auto_type = "classification"
    #         else:
    #             auto_type = "regression"

    #     print(f"\nDetected Problem Type: {auto_type} (unique y = {n_unique})")
    #     user_choice = input("Confirm detected type? (yes to accept / no to override): ").strip().lower()

    def confirm_problem_type(self) -> str:
        if self.y_raw is None:
            raise ValueError("Run preprocess_data before confirm_problem_type.")

        # Fixed regression setup (your requirement)
        self.problem_type = "regression"
        self.target_col = self.target_col or "Sales_TY"
        self.y = pd.to_numeric(self.y_raw, errors="coerce")

        return self.problem_type


    def build_final_model_matrices(self) -> None:
        if self.X_base is None or self.y is None:
            raise ValueError(
                "X_base or y not set. Run preprocess_data and confirm_problem_type first."
            )

        print("\nBuilding final model matrices (encoding + min–max)\n")

        X = self.X_base.copy()

        if self.numeric_features:
            X[self.numeric_features] = sanitize_numeric_strings(X[self.numeric_features])
            for col in self.numeric_features:
                if col not in X.columns:
                    continue
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

        if self.nominal_features:
            #X_nom = X[self.nominal_features].astype(str)
            X_nom = X[self.nominal_features].fillna("Missing").astype(str)
            #X_encoded = pd.get_dummies(X_nom, drop_first=True).astype(float)
            X_encoded = pd.get_dummies(X_nom, drop_first=False).astype(float)
        else:
            X_encoded = pd.DataFrame(index=X.index)

        if self.numeric_features:
            X_numeric = X[self.numeric_features]
        else:
            X_numeric = pd.DataFrame(index=X.index)

        X_raw = pd.concat([X_numeric, X_encoded], axis=1)
        X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)


        X_minmax = X_raw.copy()
        for col in self.numeric_features:
            if col not in X_minmax.columns:
                continue
            col_min = X_minmax[col].min()
            col_max = X_minmax[col].max()
            if pd.isna(col_min) or pd.isna(col_max) or col_max == col_min:
                X_minmax[col] = 0.0
            else:
                X_minmax[col] = (X_minmax[col] - col_min) / (col_max - col_min)

        # print("Final RAW feature matrix:\n")
        # print(X_raw.head(), "\n")

        print("Final MIN-MAX feature matrix:\n")
        print(X_minmax.head(), "\n")

        X_raw = apply_driver_prefixes(X_raw)
        X_minmax = apply_driver_prefixes(X_minmax)

        self.X_raw = X_raw
        self.X_minmax = X_minmax

        
        # final_df = X_raw.copy()
        # final_df[self.target_col] = self.y_raw.values
        # final_path = self.output_dir / "data" / "final_dataset_raw_encoded.csv"
        # self._save_table(final_df, final_path)
        # print(f"Final encoded RAW dataset saved to: {final_path}\n")

        final_mm_df = X_minmax.copy()
        final_mm_df[self.target_col] = self.y_raw.values
        final_mm_path = self.output_dir / "final_dataset_minmax.csv"
        self._save_table(final_mm_df, final_mm_path)
        print(f"Final encoded MIN-MAX dataset saved to: {final_mm_path}\n")

        logger.info(
            "Final matrices built | X_raw=%s | X_minmax=%s",
            X_raw.shape,
            X_minmax.shape,
        )


    def fit_global_model_for_explainability(self) -> None:
        if self.X_minmax is None or self.y is None or self.problem_type is None:
            raise ValueError(
                "Need X_minmax, y and problem_type. "
                "Run preprocess_data, confirm_problem_type, build_final_model_matrices first."
            )

        X = self.X_minmax
        y = self.y

        
        if self.problem_type == "regression":
            model = LinearRegression()
            model.fit(X, y)
        else:
            #model = LogisticRegression(max_iter=10000)
            model = LogisticRegression(
                solver="saga",
                penalty="l2",
                C=0.5,          # ← ADD THIS
                tol=1e-3,       # ← ADD THIS
                max_iter=3000,
                n_jobs=-1,
                random_state=42
            )
            model.fit(X, y)

        self.global_model = model
        self.logger.info(
            "Global model fitted for explainability | type=%s | n_features=%d",
            self.problem_type,
            X.shape[1],
        )


    def run_shap_explainer(self, max_rows: int = None) -> None:
        if not hasattr(self, "global_model"):
            raise ValueError("Global model not fitted. Call fit_global_model_for_explainability() first.")
        if self.X_minmax is None or self.X_raw is None:
            raise ValueError("X_minmax / X_raw not set. Run build_final_model_matrices first.")

        X_mm = self.X_minmax
        X_raw = self.X_raw
        y = self.y

        prefix = "shap"

        
        if max_rows is None:
            idx = X_mm.index
        else:
            idx = X_mm.sample(n=min(max_rows, len(X_mm)), random_state=42).index

        X_mm_sample = X_mm.loc[idx]
        # X_raw_sample = X_raw.loc[idx]

        shap_dir = self.output_dir


        
        if self.problem_type == "regression":
            explainer_mm = shap.LinearExplainer(self.global_model, X_mm_sample)
            shap_values_mm = explainer_mm.shap_values(X_mm_sample)
        else:
            explainer_mm = shap.LinearExplainer(self.global_model, X_mm_sample)
            all_shap_mm = explainer_mm.shap_values(X_mm_sample)

            if isinstance(all_shap_mm, list):
                shap_values_mm = all_shap_mm[1]
            else:
                if all_shap_mm.ndim == 3:
                    shap_values_mm = all_shap_mm[:, :, 1]
                else:
                    shap_values_mm = all_shap_mm

        
        shap_df_mm = pd.DataFrame(shap_values_mm, columns=X_mm_sample.columns)

        # REMOVE dummy / non-occurring drivers
        shap_df_mm = shap_df_mm.loc[
            :, (~shap_df_mm.columns.str.endswith("_0")) &
            (~shap_df_mm.columns.str.contains("Missing", case=False))
        ]

        self.shap_values_minmax = shap_df_mm.copy()

        shap_df_mm.insert(0, "model_name",
                  "LogisticRegression" if self.problem_type == "classification" else "LinearRegression")
        shap_df_mm.insert(1, "transformation", "min-max")
        shap_df_mm.insert(2, "target", self.target_col)
        #shap_df_mm.to_csv(shap_dir / f"{prefix}_values_minmax.csv", index=False)


        agg_mm = _build_shap_aggregate_table(
            shap_values_mm,
            feature_names=X_mm_sample.columns.tolist(),
            model_name="LogisticRegression" if self.problem_type=="classification" else "LinearRegression",
            transformation="min-max",
            target=self.target_col,
        )
        # REMOVE zero-impact SHAP rows
        # agg_mm = agg_mm[
        #     (~agg_mm["column"].str.endswith("_0")) &
        #     (~agg_mm["column"].str.contains("Missing", case=False))
        # ]


        # agg_mm = agg_mm[~agg_mm["drivers"].str.endswith("_0")]
        # agg_mm = agg_mm[~agg_mm["drivers"].str.contains("Missing", case=False, na=False)]

        agg_mm = agg_mm[~agg_mm["encoded_drivers"].str.endswith("_0")]
        agg_mm = agg_mm[~agg_mm["encoded_drivers"].str.contains("Missing", case=False, na=False)]


        self.shap_agg_minmax = agg_mm.copy()

        #agg_mm.to_csv(shap_dir / f"{prefix}_aggregate_minmax.csv", index=False)

        
        plt.figure()
        shap.summary_plot(shap_values_mm, X_mm_sample, show=False)
        plt.tight_layout()
        plt.savefig(shap_dir / f"{prefix}_summary_minmax.png", bbox_inches="tight")
        plt.close()

        
        # if self.problem_type == "regression":
        #     model_raw = LinearRegression()
        #     model_raw.fit(X_raw, y)
        #     explainer_raw = shap.LinearExplainer(model_raw, X_raw_sample)
        #     shap_values_raw = explainer_raw.shap_values(X_raw_sample)
        # else:
        #     #model_raw = LogisticRegression(max_iter=10000)
        #     model_raw = LogisticRegression(
        #         solver="saga",
        #         penalty="l2",
        #         C=0.5,          # ← ADD THIS
        #         tol=1e-3,       # ← ADD THIS
        #         max_iter=3000,
        #         n_jobs=-1,
        #         random_state=42
        #     )
        #     model_raw.fit(X_raw, y)
        #     explainer_raw = shap.LinearExplainer(model_raw, X_raw_sample)
        #     all_shap_raw = explainer_raw.shap_values(X_raw_sample)

        #     if isinstance(all_shap_raw, list):
        #         shap_values_raw = all_shap_raw[1]
        #     else:
        #         if all_shap_raw.ndim == 3:
        #             shap_values_raw = all_shap_raw[:, :, 1]
        #         else:
        #             shap_values_raw = all_shap_raw

        
        # shap_df_raw = pd.DataFrame(shap_values_raw, columns=X_raw_sample.columns)

        # # REMOVE dummy / non-occurring drivers
        # shap_df_raw = shap_df_raw.loc[
        #     :, (~shap_df_raw.columns.str.endswith("_0")) &
        #     (~shap_df_raw.columns.str.contains("Missing", case=False))
        # ]

        # shap_df_raw.insert(0, "model_name",
        #            "LogisticRegression" if self.problem_type == "classification" else "LinearRegression")
        # shap_df_raw.insert(1, "transformation", "original")
        # shap_df_raw.insert(2, "target", self.target_col)
        # shap_df_raw.to_csv(shap_dir / f"{prefix}_values.csv", index=False)
        
        # agg_raw = _build_shap_aggregate_table(
        #     shap_values_raw,
        #     feature_names=X_raw_sample.columns.tolist(),
        #     model_name="LogisticRegression" if self.problem_type=="classification" else "LinearRegression",
        #     transformation="original",
        #     target=self.target_col,
        # )
        # agg_raw = agg_raw[
        #     (~agg_raw["column"].str.endswith("_0")) &
        #     (~agg_raw["column"].str.contains("Missing", case=False))
        # ]
        # agg_raw.to_csv(shap_dir / f"{prefix}_aggregate.csv", index=False)

        
        # plt.figure()
        # shap.summary_plot(shap_values_raw, X_raw_sample, show=False)
        # plt.tight_layout()
        # plt.savefig(shap_dir / f"{prefix}_summary.png", bbox_inches="tight")
        # plt.close()

        print(f"SHAP values and summary plots saved under: {shap_dir}")
        self.logger.info(
            "SHAP explainer run complete | rows=%d | features=%d",
            X_mm_sample.shape[0],
            X_mm_sample.shape[1],
        )

    def run_bivariate_drivers(self) -> None:
        if self.X_raw is None or self.X_minmax is None or self.y is None:
            raise ValueError("Final matrices or y not set. Run build_final_model_matrices first.")
        if self.problem_type not in {"classification", "regression"}:
            raise ValueError("Problem type must be classification or regression.")

        out_dir = self.output_dir
        target_name = self.target_col or "__TARGET__"

        print("\nRunning bivariate driver analysis\n")

        if self.problem_type == "regression":
            mm_df = run_bivariate_regression(
                X_raw=self.X_raw,
                X_minmax=self.X_minmax,
                y=self.y,
                output_dir=out_dir,
                target_name=target_name,
            )

            #self.driver_results_raw = raw_df
            self.driver_results_minmax = mm_df

            #print("\nOLS Regression (RAW)")
            #print(raw_df.head(10))

            print("\nOLS Regression (MIN–MAX)")
            print(mm_df.head(10))
        else:
            print("\nRunning Logistic Regression (RAW)")
            df_log_raw = run_bivariate_logistic(
                X_raw=self.X_raw,
                y=self.y,
                output_dir=out_dir,
                target_name=self.target_col,
                file_suffix="",
                transformation_label="original",
            )

            print("\nRunning Logistic Regression (MIN–MAX)")
            df_log_mm = run_bivariate_logistic(
                X_raw=self.X_minmax,
                y=self.y,
                output_dir=out_dir,
                target_name=self.target_col,
                file_suffix="_minmax",
                transformation_label="min-max",
            )

            self.driver_results_logistic_raw = df_log_raw
            self.driver_results_logistic_minmax = df_log_mm

            print("\nLogistic Regression (RAW) top rows:")
            print(df_log_raw.head(10))

            print("\nLogistic Regression (MIN–MAX) top rows:")
            print(df_log_mm.head(10))


    def summarize(self) -> None:
        print("\n=== Final Summary ===")
        print(f"Run ID      : {self.run_id}")
        print(f"Problem Type: {self.problem_type}")
        print(f"Target      : {self.target_col}")
        print("Outputs are available under:")
        print(f" -> {self.output_dir}")
