import pandas as pd
import glob
import plotly.graph_objects as go
import argparse
import plotly.io as pio

pio.renderers.default = "browser"


def main():
    parser = argparse.ArgumentParser(description="Plot radar chart for models")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Specify a category (parent) to select all its metrics.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Comma-separated list of metrics names to select directly.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./output_csvs",
        help="Directory containing CSV files.",
    )
    parser.add_argument(
        "--save_html",
        help="whether save the generated html file, please do so on slurm",
        default=True,
        type=bool,
    )
    args = parser.parse_args()

    if args.category is None and args.metrics is None:
        raise ValueError("You must specify either --category or --metrics.")
    if args.category and args.metrics:
        raise ValueError(
            "Please specify only one of --category or --metrics, not both."
        )

    csv_files = glob.glob(f"{args.data_dir}/*.csv")

    model_data = {}  # {model_name: [metric1_mean, metric2_mean, ...]}
    metrics_list = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        if args.category:
            metrics_df = df[df["parent"] == args.category]
            selected_metrics = metrics_df["name"].tolist()
            values = metrics_df[["name", "mean"]].dropna()
        else:
            selected_metrics = [m.strip() for m in args.metrics.split(",")]
            values = df[df["name"].isin(selected_metrics)][["name", "mean"]].dropna()

        mean_values = []
        for metric_name in selected_metrics:
            row = values[values["name"] == metric_name]
            if not row.empty:
                mean_values.append(float(row["mean"].values[0]))
            else:
                mean_values.append(0.0)

        model_name = csv_file.split("/")[-1].replace(".csv", "")
        model_data[model_name] = mean_values
        metrics_list = selected_metrics

    fig = go.Figure()

    for model_name, values in model_data.items():
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],  # close loop for the ladar chart
                theta=metrics_list + [metrics_list[0]],
                fill="toself",
                name=model_name,
            )
        )

    fig.update_layout(
        title=f"Radar Chart for {'Category: ' + args.category if args.category else 'Metrics: ' + args.metrics}",
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=True,
    )

    if args.save_html:
        fig.write_html("radar_chart.html")
        print(
            f"results saved to radar_chart.html, please download and open in local browser."
        )
    else:
        fig.show()


if __name__ == "__main__":
    main()
