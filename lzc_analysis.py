import argparse
import os
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats

METHOD = "khodadad2018_200Hz_120s_all"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default=METHOD)
    return parser.parse_args()

args = parse_args()

# Read the Excel file
excel_file_name = 'N2O-study2-data-for-Luka.xlsx'
excel_sheet_name = 'Sheet1'
excel_file_path = os.path.join("data_raw", excel_file_name)
excel_df = pd.read_excel(excel_file_path, sheet_name=excel_sheet_name)
excel_df = excel_df.sort_values(by=['ID', 'Session'])

# Read the CSV file
csv_file_name = f'lzc_{args.method}_file_cond.csv'
csv_file_path = os.path.join(f"data_{args.method}_lzc", csv_file_name)
csv_df = pd.read_csv(csv_file_path)
csv_df = csv_df.rename(columns={'session': 'phase'})
csv_df['ID'] = csv_df['file'].str[:4]
csv_df['Session'] = csv_df['file'].str[5].astype(int)
csv_df = csv_df.drop(columns=['file'])

df = pd.merge(excel_df, csv_df, on=['ID', 'Session'], how='inner')
df = df[['ID', 'Session', 'phase'] + [col for col in df.columns if col not in ['ID', 'Session', 'phase']]]

# print()
# print(excel_df)
# print(csv_df)
# print()
# print(df)
# print()

#print all df columns
# print(f"DataFrame columns: {list(df.columns)}\n")

# Print average complexity for different conditions
print("\n" + "=" * 80)
print("T-TEST ANALYSIS")
print("=" * 80)

pre = df[df['phase'] == 'pre']['lzc_mean']
print(f"Mean LZC (PRE): {pre.mean():.4f} ± {pre.std():.4f} ({pre.count()} subject-session pairs)")

air = df[(df['phase'] == 'dur') & (df['Drug'] == 'Medical Air')]['lzc_mean']
print(f"Mean LZC (AIR): {air.mean():.4f} ± {air.std():.4f} ({air.count()} subject-session pairs)")

n2o = df[(df['phase'] == 'dur') & (df['Drug'] == 'N2O')]['lzc_mean']
print(f"Mean LZC (N2O): {n2o.mean():.4f} ± {n2o.std():.4f} ({n2o.count()} subject-session pairs)")

print()

pre_vals = pre.dropna()
air_vals = air.dropna()
n2o_vals = n2o.dropna()

t_stat, p_val = stats.ttest_ind(air_vals, pre_vals, equal_var=False)
print(f"p-value (AIR vs PRE, Welch t-test): {p_val:.7f} (t={t_stat:.3f}) - {'NOT ' if p_val >= 0.05 else ''}SIGNIFICANT")

t_stat, p_val = stats.ttest_ind(n2o_vals, pre_vals, equal_var=False)
print(f"p-value (N2O vs PRE, Welch t-test): {p_val:.7f} (t={t_stat:.3f}) - {'NOT ' if p_val >= 0.05 else ''}SIGNIFICANT")

t_stat, p_val = stats.ttest_ind(n2o_vals, air_vals, equal_var=False)
print(f"p-value (N2O vs AIR, Welch t-test): {p_val:.7f} (t={t_stat:.3f}) - {'NOT ' if p_val >= 0.05 else ''}SIGNIFICANT")

print("=" * 80 + "\n")

# ROBUST LINEAR MODEL

df['intercept'] = 1
df['gender'] = df['Gender'].map({'M': 0, 'F': 1})
# df['condition'] = pd.NA
# df.loc[df['phase'] == 'pre', 'condition'] = 'pre'
# df.loc[(df['phase'] == 'dur') & (df['Drug'] == 'Medical Air'), 'condition'] = 'air'
# df.loc[(df['phase'] == 'dur') & (df['Drug'] == 'N2O'), 'condition'] = 'n2o'

# variables = ['intercept', 'depressed', 'BDI', 'BDI_Anh', 'BDI_Mel', 'TAI', 'gender', 'age', 'age_squared']
# ID, Session, Drug, Order, dur_lzc_mean, pre_lzc_mean, dur_lzc_std, pre_lzc_std, dur_epoch_count, pre_epoch_count

# Perform regression analysis for each combination of drug and phase
def perform_rlm_per_drug_phase(df, target):
    variables = ['intercept', 'gender', 'T2_DEQ_feel', 'T2_DEQ_high', 'T2_DEQ_dislike', 'T2_DEQ_like', 'T2_DEQ_want', 'T2_baes_stim', 'T2_baes_sed', 'T2_cadss_total', 'T2_psi_total', 'delta_deq_feel', 'delta_deq_high', 'delta_deq_like', 'delta_deq_dislike', 'delta_deq_want', 'delta_cadss_total', 'delta_psi_total', 'delta_baes_stim', 'delta_baes_sed', 'delta_alc_drinks', 'delta_alc_sim', 'ASC_total', 'no.side.eff.endorsed', 'total.no.side.eff.possible', 'side.eff.avg.severity', 'weighted.severity']
    rlm_results = pd.DataFrame(columns=['drug', 'phase', 'variable', 'p_value', 't_value', 'p_significant', 't_significant'])

    for drug in ['Medical Air', 'N2O']:
        for phase in ['pre', 'dur']:
            # Run sm.RLM for each subset of the DataFrame
            subset = df[(df['Drug'] == drug) & (df['phase'] == phase)]
            if subset.empty:
                continue
            model = sm.RLM(subset[target], subset[variables], missing='drop', M=sm.robust.norms.HuberT())
            results = model.fit()
            # print(results.summary())
            for var in variables[1:]:
                p_significant = results.pvalues[var] <= 0.05
                t_significant = abs(results.tvalues[var]) >= 1.7
                new_permutation = pd.DataFrame([{
                    'drug': drug,
                    'phase': phase,
                    'variable': var,
                    'p_value': results.pvalues[var],
                    't_value': results.tvalues[var],
                    'p_significant': p_significant,
                    't_significant': t_significant
                }])
                rlm_results = pd.concat([rlm_results, new_permutation], ignore_index=True)
    
    return rlm_results


# Perform regression analysis for each phase with drug included as a variable
def perform_rlm_per_phase(df, target):
    df['drug'] = df['Drug'].map({'Medical Air': 0, 'N2O': 1})
    variables = ['intercept', 'gender', 'drug', 'T2_DEQ_feel', 'T2_DEQ_high', 'T2_DEQ_dislike', 'T2_DEQ_like', 'T2_DEQ_want', 'T2_baes_stim', 'T2_baes_sed', 'T2_cadss_total', 'T2_psi_total', 'delta_deq_feel', 'delta_deq_high', 'delta_deq_like', 'delta_deq_dislike', 'delta_deq_want', 'delta_cadss_total', 'delta_psi_total', 'delta_baes_stim', 'delta_baes_sed', 'delta_alc_drinks', 'delta_alc_sim', 'ASC_total', 'no.side.eff.endorsed', 'total.no.side.eff.possible', 'side.eff.avg.severity', 'weighted.severity']
    rlm_results = pd.DataFrame(columns=['phase', 'variable', 'p_value', 't_value', 'p_significant', 't_significant'])

    for phase in ['pre', 'dur']:
        # Run sm.RLM for each subset of the DataFrame
        subset = df[df['phase'] == phase]
        if subset.empty:
            continue
        model = sm.RLM(subset[target], subset[variables], missing='drop', M=sm.robust.norms.HuberT())
        results = model.fit()
        # print(results.summary())
        for var in variables[1:]:
            p_significant = results.pvalues[var] <= 0.05
            t_significant = abs(results.tvalues[var]) >= 1.7
            new_permutation = pd.DataFrame([{
                'phase': phase,
                'variable': var,
                'p_value': results.pvalues[var],
                't_value': results.tvalues[var],
                'p_significant': p_significant,
                't_significant': t_significant
            }])
            rlm_results = pd.concat([rlm_results, new_permutation], ignore_index=True)
    
    return rlm_results


# Perform regression analysis for each phase with drug included as a variable
def perform_rlm(df, target):
    df['drug'] = df['Drug'].map({'Medical Air': 0, 'N2O': 1})
    df['phase'] = df['phase'].map({'pre': 0, 'dur': 1})
    variables = ['intercept', 'gender', 'drug', 'phase', 'T2_DEQ_feel', 'T2_DEQ_high', 'T2_DEQ_dislike', 'T2_DEQ_like', 'T2_DEQ_want', 'T2_baes_stim', 'T2_baes_sed', 'T2_cadss_total', 'T2_psi_total', 'delta_deq_feel', 'delta_deq_high', 'delta_deq_like', 'delta_deq_dislike', 'delta_deq_want', 'delta_cadss_total', 'delta_psi_total', 'delta_baes_stim', 'delta_baes_sed', 'delta_alc_drinks', 'delta_alc_sim', 'ASC_total', 'no.side.eff.endorsed', 'total.no.side.eff.possible', 'side.eff.avg.severity', 'weighted.severity']
    rlm_results = pd.DataFrame(columns=['variable', 'p_value', 't_value', 'p_significant', 't_significant'])

    model = sm.RLM(df[target], df[variables], missing='drop', M=sm.robust.norms.HuberT())
    results = model.fit()
    # print(results.summary())
    for var in variables[1:]:
        p_significant = results.pvalues[var] <= 0.05
        t_significant = abs(results.tvalues[var]) >= 1.7
        new_permutation = pd.DataFrame([{
            'variable': var,
            'p_value': results.pvalues[var],
            't_value': results.tvalues[var],
            'p_significant': p_significant,
            't_significant': t_significant
        }])
        rlm_results = pd.concat([rlm_results, new_permutation], ignore_index=True)
    
    return rlm_results


rlm_per_drug_phase = perform_rlm_per_drug_phase(df, 'lzc_mean')
rlm_per_phase = perform_rlm_per_phase(df, 'lzc_mean')
rlm = perform_rlm(df, 'lzc_mean')

# print("\n" + "=" * 100)
# print("RLM Analysis")
# print("=" * 100)

# print("\nRLM Results (Per Drug Per Phase):")
# print(rlm_per_drug_phase.to_string(index=False))
# rlm_per_drug_phase.to_excel(os.path.join(f"data_{args.method}_lzc", f"rlm_{args.method}_per_drug_phase.xlsx"), index=False)

# print("\nRLM Results (Per Phase):")
# print(rlm_per_phase.to_string(index=False))
# rlm_per_phase.to_excel(os.path.join(f"data_{args.method}_lzc", f"rlm_{args.method}_per_phase.xlsx"), index=False)

# print("\nRLM Results:")
# print(rlm.to_string(index=False))
# rlm.to_excel(os.path.join(f"data_{args.method}_lzc", f"rlm_{args.method}.xlsx"), index=False)

print("\n" + "=" * 100)

# print only significant results
# print("\nSignificant RLM Results (Per Drug Per Phase):")
# print(rlm_per_drug_phase[rlm_per_drug_phase['p_significant'] | rlm_per_drug_phase['t_significant']].to_string(index=False))

# print("\nSignificant RLM Results (Per Phase):")
# print(rlm_per_phase[rlm_per_phase['p_significant'] | rlm_per_phase['t_significant']].to_string(index=False))

print("\nSignificant RLM Results:")
print(rlm[rlm['p_significant'] | rlm['t_significant']].to_string(index=False))

# print("\n" + "=" * 100)

# Save results to CSV
output_csv_path = os.path.join(f"data_{args.method}_lzc", f"rlm_{args.method}_per_drug_phase.csv")
rlm_per_drug_phase.to_csv(output_csv_path, index=False)

output_csv_path = os.path.join(f"data_{args.method}_lzc", f"rlm_{args.method}_per_phase.csv")
rlm_per_phase.to_csv(output_csv_path, index=False)

output_csv_path = os.path.join(f"data_{args.method}_lzc", f"rlm_{args.method}.csv")
rlm.to_csv(output_csv_path, index=False)

# print("\n" + "=" * 100)

# print(f"\nRLM results saved to data_{args.method}_lzc folder\n")
