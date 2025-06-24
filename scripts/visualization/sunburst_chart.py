import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import argparse

pio.renderers.default = "browser"

def plot_interactive_sunburst_chart(df, save_html=True):
    """
    Args:
        - df: the dataframe loaded from csv file.
        - save_html: if cannot directly forward port, then download and open in local browser.
    """
    fig = px.sunburst(
        df,
        names="name",
        parents="parent",
        values="value",
        # color="color_group",  #  use color_group, or else will use the default color map
        hover_data=["mean", "std"],  # additional info
        branchvalues="total",  # as the total value
        title="Versa Metrics Interactive Sunburst Graph",
    )

    # Turn off 'value' in hover text
    fig.update_traces(
        # text=df["description1"] + " | " + df["description2"],
        text=df["mean"],
        textinfo="label+text",  # or "label+percent parent" if you want percent on wedges
        hovertemplate=(
            "%{label}<br>"
            "Mean: %{customdata[0]}<br>"
            "Std: %{customdata[1]}<extra></extra>"
        )
    )
    if save_html:
        fig.write_html("sunburst_chart.html")
        print(f'results saved to sunburst_chart.html, please download and open in local browser.')
    else:
        fig.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_filepath", help="the path of result file",
                        default="scripts/visualization/base_category.csv", type=str)
    parser.add_argument("--save_html", help="whether save the generated html file, please do so on slurm",
                        default=True, type=bool)
    args = parser.parse_args()

    df = pd.read_csv(args.result_filepath)
    save_html = args.save_html

    plot_interactive_sunburst_chart(df, save_html)


if __name__=="__main__":
    main()

