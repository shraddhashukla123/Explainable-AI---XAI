# import os
# import pandas as pd
# from xai_runner.runner import ExplainableModelRunner
# from xai_runner.config import DATA_DIR


# def auto_select_dataset() -> str:
#     files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
#     if not files:
#         raise FileNotFoundError(
#             f"No CSV files found in DATA_DIR: {DATA_DIR}. "
#             "Please add at least one .csv file."
#         )

#     files.sort()
#     selected = files[0]
#     full_path = os.path.join(DATA_DIR, selected)

#     print("Explainable Driver Engine")
#     print(f"Dataset Found : {selected}")
#     print(f"Path          : {full_path}\n")

#     return full_path


# def main() -> None:
#     try:
#         dataset_path = auto_select_dataset()
#     except FileNotFoundError as e:
#         print("\nERROR:", str(e))
#         return

#     df = pd.read_csv(dataset_path)
#     print(f"Loaded Dataset | Shape = {df.shape}\n")

#     runner = ExplainableModelRunner(df)

#     runner.select_target_column()
#     runner.detect_feature_types()
#     runner.preprocess_data()
#     runner.confirm_problem_type()
#     runner.build_final_model_matrices()
#     runner.run_bivariate_drivers()
#     runner.fit_global_model_for_explainability()
#     runner.run_shap_explainer(max_rows=None)
#     runner.summarize()

#     print("\nPipeline complete.\n")


# if __name__ == "__main__":
#     main()







import os
import pandas as pd
from pathlib import Path
from xai_runner.runner import ExplainableModelRunner
from xai_runner.config import DATA_DIR, OUTPUT_DIR



def main() -> None:
    dataset_path = os.path.join(DATA_DIR, "dataset.csv")
    df = pd.read_csv(dataset_path)
    print(f"Loaded Dataset | Shape = {df.shape}\n")

    records = int(df.shape[0])
    columns = int(df.shape[1])

    dataset_name = "dataset"
    model_name = "LinearRegression"

    data_description_text = (
        "This dataset is used for demand forecasting and driver analysis. "
        "It contains sales, pricing, macroeconomic, weather, event, and "
        "calendar-related features at weekly granularity."
    )

    about_explainable_ai = (
        "Explainable AI (XAI) in predictive modeling provides clear, human-"
        "understandable reasons behind a model’s predictions, moving beyond "
        "black-box approaches to explain why a decision was made, not just "
        "what the outcome is."
    )

    about_dashboard = (
        "1. Estimation of Impact – % change in KPI if 1% change in KBD\n"
        "2. Splitting Relative Importance of business drivers against "
        "controllable vs. uncontrollable factors\n"
        "3. Rank-ordering the KBDs in terms of impact, significance and importance"
    )

    meta_df = pd.DataFrame([{
        "Dataset": dataset_name,
        "Problem": "Regression",
        "Target": "Sales_TY",
        "DataDescription": data_description_text,
        "model_name": model_name,
        "transformation": "min-max",
        "about_xai": about_explainable_ai,
        "about_dashboard": about_dashboard,
        "records": records,
        "columns": columns,
    }])

    meta_path = Path(OUTPUT_DIR) / "data_description.csv"
    meta_df.to_csv(meta_path, index=False, encoding="utf-8-sig")
    print(f"Saved global data_description.csv: {meta_path}\n")


    # Collect appended outputs
    all_driver_results = []
    all_shap_agg_results = []
    all_shap_value_results = []

    regions = df["Region"].dropna().unique()

    for region in regions:
        df_region = df[df["Region"] == region]

        categories = df_region["category_area"].dropna().unique()

        for category in categories:
            df_cat = df_region[df_region["category_area"] == category]

            products = df_cat["Product"].dropna().unique()

            print(f"Starting Category : {category}")
            print(f"Products Found    : {len(products)}")

            for product in products:
                df_prod = df_cat[df_cat["Product"] == product]

                if df_prod.shape[0] < 20:
                    print(f"Skipping Product: {product} (insufficient data)\n")
                    continue

                print(f"Running XAI | Category: {category} | Product: {product}")


                runner = ExplainableModelRunner(df_prod)
                runner.output_dir = Path(OUTPUT_DIR)
                runner.meta = {
                    "region": region,
                    "category": category,
                    "product": product
                }
                #runner.output_dir = run_output_dir

                runner.target_col = "Sales_TY"
                runner.problem_type = "regression"

                runner.detect_feature_types()
                runner.preprocess_data()
                runner.confirm_problem_type()
                runner.build_final_model_matrices()

                runner.run_bivariate_drivers()
                runner.fit_global_model_for_explainability()
                runner.run_shap_explainer(max_rows=None)
                runner.summarize()

                #driver_path = Path(OUTPUT_DIR) / "driver_analysis_coefficients_minmax.csv"

                #driver_df = pd.read_csv(driver_path)
                driver_df = runner.driver_results_minmax.copy()
                driver_df.insert(0, "Region", region)
                driver_df.insert(1, "category_area", category)
                driver_df.insert(2, "Product", product)
                all_driver_results.append(driver_df)

                #shap_agg_path = Path(OUTPUT_DIR) / "shap_aggregate_minmax.csv"

                #shap_agg_df = pd.read_csv(shap_agg_path)
                shap_agg_df = runner.shap_agg_minmax.copy()
                shap_agg_df.insert(0, "Region", region)
                shap_agg_df.insert(1, "category_area", category)
                shap_agg_df.insert(2, "Product", product)
                all_shap_agg_results.append(shap_agg_df)

                #shap_values_path = Path(OUTPUT_DIR) / "shap_values_minmax.csv"

                #shap_values_df = pd.read_csv(shap_values_path)
                shap_values_df = runner.shap_values_minmax.copy()
                shap_values_df.insert(0, "Region", region)
                shap_values_df.insert(1, "category_area", category)
                shap_values_df.insert(2, "Product", product)
                all_shap_value_results.append(shap_values_df)

                print(f"Completed | Category: {category} | Product: {product}\n")

    if not all_driver_results:
        print("No driver analysis outputs were generated.")
        return

    if not all_shap_agg_results:
        print("No SHAP aggregate outputs were generated.")
        return

    if not all_shap_value_results:
        print("No SHAP values outputs were generated.")
        return

    final_driver_df = pd.concat(all_driver_results, ignore_index=True)
    final_driver_df.to_csv(
        Path(OUTPUT_DIR) / "Driver_Analysis_Coefficients.csv",
        index=False
    )

    final_shap_agg_df = pd.concat(all_shap_agg_results, ignore_index=True)
    final_shap_agg_df.to_csv(
        Path(OUTPUT_DIR) / "Shap_Aggregate.csv",
        index=False
    )

    final_shap_values_df = pd.concat(all_shap_value_results, ignore_index=True)
    final_shap_values_df.to_csv(
        Path(OUTPUT_DIR) / "Shap_Values.csv",
        index=False
    )

    print("Final appended XAI outputs created successfully.")
    print("\nAll runs completed successfully.\n")


if __name__ == "__main__":
    main()
