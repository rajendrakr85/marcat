import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations, permutations
from collections import defaultdict
from math import factorial


def subsets(S):
    '''Returns all possible subsets of the given set'''
    s = []
    for i in range(1, len(S) + 1):
        s.extend(map(list, combinations(S, i)))
    return list(map('+'.join, s))


class channel_attributions():
    def data_preprocessing(self, data_df):
        start = datetime.now()
        is_coverted_df = pd.DataFrame(data_df.groupby('channel_name').sum().reset_index())
        paths_lis = []
        #visit_id_lis = []
        first_touch_pnt = []
        last_touch_pnt = []
        for data_tupl, is_converted in zip(data_df.groupby('channel_name'),
                                           is_coverted_df["is_converted"]):
            channel_lis = data_tupl[1].unique().tolist()
            if is_converted:
                conv = "Conversion"
            else:
                conv = "NULL"
            #visit_id_lis.append(data_tupl[0])
            first_touch_pnt.append(channel_lis[0])
            last_touch_pnt.append(channel_lis[-1])
            path = "Start > " + " > ".join(channel_lis) + " > " + conv
            paths_lis.append(path)

        test_df = pd.DataFrame()
        #test_df["visitor_id"] = visit_id_lis
        test_df["Paths_journey"] = paths_lis
        test_df["First Touch"] = first_touch_pnt
        test_df["Last Touch"] = last_touch_pnt
        #test_df["is_converted"] = is_coverted_df["is_converted"]
        test_df["Paths"] = test_df["Paths_journey"].str.replace("Start > ", "")
        test_df["Paths"] = test_df["Paths"].str.replace(" > NULL", "")
        test_df["Paths"] = test_df["Paths"].str.replace(" > Conversion", "")
        test_df["Paths_len"] = test_df["Paths"].str.split(" > ").str.len()
        #test_df[["visitor_id", "Paths_journey"]].to_csv("results/Customer_journeys.csv", index=False)
        test_df[["visitor_id", "Paths_journey"]].to_csv("results/Customer_journeys.csv", index=False)

        end = datetime.now()
        diff = start - end

        # print("Processed Time : ", diff // (60 * 60))

        return test_df

    def marko_chain(self, data):
        df1 = data["Paths_journey"].str.split(">", expand=True)

        df_new = pd.DataFrame()
        for n in range(1, len(df1.columns)):
            temp_df = pd.crosstab(df1[n - 1], df1[n])
            df_new = pd.concat([df_new, temp_df])
        df_grp = df_new.groupby(df_new.index).sum()
        df_grp = df_grp.reset_index()

        df_null = {'index': 'NULL'}
        df_conv = {'index': 'Conversion'}

        df_grp = df_grp.append(df_null, ignore_index=True)
        df_grp = df_grp.append(df_conv, ignore_index=True)

        df_grp["Start"] = 0.0

        df_grp.fillna(0.0, inplace=True)
        df_grp.columns = df_grp.columns.str.strip()
        df_grp["index"] = df_grp["index"].str.strip()

        # mt_grp.loc[mt_grp[mt_grp["index"]=="start"].index,"start"]=1.0
        df_grp.loc[df_grp[df_grp["index"] == "NULL"].index, "NULL"] = 1.0
        df_grp.loc[df_grp[df_grp["index"] == "Conversion"].index, "Conversion"] = 1.0

        df_grp = df_grp.set_index("index")

        df_grp_prob = df_grp.div(df_grp.sum(axis=1).values, axis=0)
        # print("END : ", datetime.now())
        return df_grp_prob

    def removal_effects(self, df, conversion_rate):
        removal_effects_dict = {}
        channels = [channel for channel in df.columns if channel not in ['Start',
                                                                         'NULL',
                                                                         'Conversion']]
        for channel in channels:
            removal_df = df.drop(channel, axis=1).drop(channel, axis=0)
            for column in removal_df.columns:
                row_sum = np.sum(list(removal_df.loc[column]))
                null_pct = float(1) - row_sum
                if null_pct != 0:
                    removal_df.loc[column]['NULL'] = null_pct
                removal_df.loc['NULL']['NULL'] = 1.0

            removal_to_conv = removal_df[
                ['NULL', 'Conversion']].drop(['NULL', 'Conversion'], axis=0)
            removal_to_non_conv = removal_df.drop(
                ['NULL', 'Conversion'], axis=1).drop(['NULL', 'Conversion'], axis=0)

            removal_inv_diff = np.linalg.inv(
                np.identity(
                    len(removal_to_non_conv.columns)) - np.asarray(removal_to_non_conv))
            removal_dot_prod = np.dot(removal_inv_diff, np.asarray(removal_to_conv))
            removal_cvr = pd.DataFrame(removal_dot_prod,
                                       index=removal_to_conv.index)[[1]].loc['Start'].values[0]
            removal_effect = 1 - removal_cvr / conversion_rate
            removal_effects_dict[channel] = removal_effect

        return removal_effects_dict

    def markov_chain_allocations(self, removal_effects, total_conversions):
        re_sum = np.sum(list(removal_effects.values()))

        return {k: (v / re_sum) * total_conversions for k, v in removal_effects.items()}

    def shapley_value(self, test_df, unique_channels):
        N = sorted(unique_channels)
        coalitions = subsets(N)
        coalitions_lbl = ['S{}'.format(i) for i in range(1, len(coalitions) + 1)]

        df_new = test_df.groupby(["Paths"]).sum().reset_index()
        df_new["Paths_len"] = df_new["Paths"].str.split(" > ").str.len()
        df_new.sort_values("Paths_len").reset_index(drop=True, inplace=True)
        df_new["conversion_ratio"] = df_new["is_converted"] / df_new["is_converted"].sum()
        df_new["Paths"] = df_new["Paths"].str.replace(" > ", "+")
        df_new = df_new.sort_values("Paths_len").reset_index(drop=True)

        lis = []
        for i in range(len(df_new)):
            lis.append('+'.join(sorted(df_new["Paths"].loc[i].split("+"))))
        df_new["Sorted_path"] = lis

        sort_df = pd.DataFrame(df_new.groupby(["Sorted_path"])["conversion_ratio"].sum().reset_index())

        ratios = []
        for i in coalitions:
            try:
                ratios.append(list(sort_df[sort_df["Sorted_path"] == i]["conversion_ratio"])[0])
            except:
                ratios.append(0)

        IR = np.array(ratios)

        d = 2 ** len(N) - 1
        B = np.matrix(np.zeros((d, d)))

        for i in range(0, d):
            A = coalitions[i]
            S = subsets(A.split('+'))
            coef = [1 if c in S else 0 for c in coalitions]
            B[i] = coef

        vS = np.dot(B, IR)
        vS = np.squeeze(np.asarray(vS))

        vSx = ['v({})'.format(lbl) for lbl in coalitions_lbl]

        shapley = defaultdict(int)
        n = len(N)
        for i in N:
            for A in coalitions:
                S = A.split('+')
                if i not in S:
                    k = len(S)  # Cardinality of set |S|
                    Si = S
                    Si.append(i)
                    Si = '+'.join(sorted(Si))
                    # Weight = |S|!(n-|S|-1)!/n!
                    weight = (factorial(k) * factorial(n - k - 1)) / factorial(n)
                    # Marginal contribution = v(S U {i})-v(S)
                    contrib = vS[coalitions.index(Si)] - vS[coalitions.index(A)]
                    shapley[i] += weight * contrib
            shapley[i] += vS[coalitions.index(i)] / n

        pd.options.display.float_format = '{:,.3f}'.format
        res = pd.DataFrame({
            'Shapley value': list(shapley.values())
        }, index=list(shapley.keys()))
        res = res.reset_index()
        res.columns = ["Channel", "Values"]
        return res
