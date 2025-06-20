import pandas as pd
import argparse

def write_vis_csv(input_file, output_file):
    # 1. Read the original metrics CSV
    df = pd.read_csv(input_file)

    # 2. Group by 'category' NOTE(yiwen) value: number of metrics in each category
    grouped = df.groupby('category').agg(
        value=('metric_name', 'count'),
        mean=('mean', 'mean'),
        std=('std', 'mean')
    ).reset_index()

    # 3. Create root node
    total_value = grouped['value'].sum()
    rows = []
    rows.append({
        'name': 'Versa Metrics',
        'parent': '',
        'value': total_value,
        'mean': '',
        'std': ''
    })

    # 4. Add category nodes
    for _, row in grouped.iterrows():
        rows.append({
            'name': row['category'],
            'parent': 'Versa Metrics',
            'value': row['value'],
            'mean': '',  
            'std': ''    
        })

    # 5. Add leaf metric nodes
    for _, row in df.iterrows():
        rows.append({
            'name': row['metric_name'],
            'parent': row['category'],
            'value': 1,
            'mean': round(row['mean'], 4),
            'std': round(row['std'], 4)
        })

    # 6. Save to new CSV
    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_file, index=False)

    print("New CSV saved as metrics_tree.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="the path of text result csv file",
                        default="metrics_analysis.csv", type=str)
    parser.add_argument("--output_file", help="the new format csv file for visualization",
                        default="metrics_tree.csv", type=str)
    args = parser.parse_args()

    write_vis_csv(args.input_file, args.output_file)

if __name__=='__main__':
    main()