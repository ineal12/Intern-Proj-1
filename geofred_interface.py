# Python 3.7.7
# adam note:
#   This script has been modified from its original state to work with other GeoFRED data.
#
#   For quickstart use, jump to 'def main' and add any additional Series IDs of interest.
#   Then, jump to 'my_path' and change that to a convenient location on your computer.
#   Finally, run the script! :)

# standard library
import math
# third-party
import pandas as pd
import datapungi_fed as dpf


class Wrangler(object):

    @staticmethod
    def annualize(dataframe):
        print("Annualizing monthly data...")
        tmp = dataframe[['year', 'county', 'feature', 'value']]
        tmp = tmp.groupby(['county', 'feature', 'year'])
        tmp = tmp.aggregate(func='mean')

        return tmp.reset_index()

    @staticmethod
    def long_to_wide(dataframe):
        print("Transforming long format to wide format...")
        tmp = dataframe.reset_index(drop=True)
        tmp = tmp.pivot_table(
            index=['year', 'county'],
            values='value',
            columns='feature'
        )

        return tmp.reset_index()

    @staticmethod
    def replace_nan(dataframe, median=False):
        """Replaces remaining NaNs with average/median values.

        Args:
            dataframe (pd.DataFrame): to transform
            median (list[str]): list of dataframe cols to median replace

        Returns: pd.DataFrame with replaced nan values
        """
        tmp = dataframe

        if median:
            tmp_med = tmp[median]
            tmp_med = tmp_med.fillna(tmp_med.median())
        
        tmp = tmp.fillna(tmp.mean())

        if median:
            tmp[tmp_med.columns] = tmp_med

        return tmp

    @staticmethod
    def normalize_features(dataframe):
        """Scales all numeric features from 0 to 1."""
        print("Normalizing feature matrix...")
        tmp = dataframe
        feats = tmp.drop(columns=['year', 'county'])
        fmax = feats.max()
        fmin = feats.min() 
        # normalize the feature matrix
        feats = (feats - fmin) / (fmax - fmin)
        tmp[feats.columns] = feats

        return tmp

    @staticmethod
    def reduce_to_hpi(dataframe):
        print("Reducing dataframe to counties with recorded HPI...")
        tmp = dataframe
        print("  > Original shape:", tmp.shape)
        tmp = tmp[tmp['All-Transactions House Price Index'].isna() == False]
        print("  > Reduced shape:", tmp.shape)

        return tmp

    @staticmethod
    def transform_fresh_columns(dataframe):
        tmp = dataframe
        tmp['value'] = tmp['value'].astype(float)
        tmp['year'] = tmp['_date'].str[:4]
        tmp = tmp.rename(columns={'series_id': 'feature', 'region': 'county'})
        # tmp = tmp[tmp['year'] == '2017']

        return tmp[['year', 'county', 'feature', 'value']]

    @staticmethod
    def drop_years(dataframe, start, end):
        """Subsets original data based on a start/end year.

        Args:
            dataframe (pd.DataFrame): dataframe to subset
            start (int): start year
            end (int): end year
        Returns: pd.DataFrame
        """
        tmp = dataframe
        tmp = tmp[(start <= tmp['year'].astype(int)) & (tmp['year'].astype(int) <= end)]

        return tmp


def pull_data_from_geodb(geodb, sids, verbose=True):
    """Gets data from the GeoFRED database.

    Pulls all the data into two dataframes based
    on whether the features are annual or monthly.

    Args:
        geodb: database object from datapungi_fed
        sids (list[str]): all of the series ids for the features we want

    Returns: tuple[pd.DataFrame] of the form (df_annual, df_month)
    """
    # cutoff_start = pd.to_datetime('2009-01-01').date()  # start after mortgage crisis
    # cutoff_end = pd.to_datetime('2017-12-31').date()  # last 'day' we have for hpi

    df_a = pd.DataFrame()  # for annual features
    df_m = pd.DataFrame()  # for monthly features

    # gets meta data for each feature from its series id, 
    # then uses meta data to pull all of the feature's regional data
    for sid in sids:
        print(f"Series ID: {sid}")

        print("  > Collecting meta data...")
        meta = geodb['meta'](series_id=sid).iloc[0, :]

        for col in ['min_date', 'max_date']:
            meta[col] = pd.to_datetime(meta[col]).date()

        print("  > Pulling dataframe...")
        data = geodb['data'](
            series_group=meta['series_group'],
            date=meta['max_date'],
            start_date=meta['min_date'],
            region_type=meta['region_type'],
            units=meta['units'],
            frequency=meta['frequency'],
            season=meta['season']
        )
        # swap series id for its feature name (change colname later)
        data['series_id'] = meta['title']

        if verbose:
            print(f"  > Title: {meta['title']}")
            print(f"  > Min date: {meta['min_date']}")
            print(f"  > Max date: {meta['max_date']}")
            print(f"  > Units: {meta['units']}")
            print(f"  > Season: {meta['season']}")
            print(f"  > Frequency: {meta['frequency']}")
            print(f"  > Region type: {meta['region_type']}")


        print(f"  > Appending to df_{meta['frequency']}...")
        if meta['frequency'] == 'a':
            df_a = df_a.append(data)
        else:
            df_m = df_m.append(data)
        continue

    df_a = Wrangler.transform_fresh_columns(df_a)
    df_m = Wrangler.transform_fresh_columns(df_m)

    return df_a, df_m


def main():
    api_key = "f125f4c130e61d9f4ad5874aadfe07ff"
    geodb = dpf.data(api_key).geo

    # equifax subprime data goes through 2020, and is quarterly
    # all annual features, no monthly yet
    # most of our features goes through 2018
    series_ids = [
        "EQFXSUBPRIME036061",  # equifax subprime
        "GDPALL17031",  # GDP all industries
        "PPAAKY21217A156NCEN",  # Estimated Percent of People of All Ages in Poverty
        "CBR21189KYA647NCEN",  # SNAP Benefits recipients
        "RACEDISPARITY041047",  # Segregation/Integration
        "DP04ACS031031",  # Burdened Households (>30% of their income on their house)
        "B14005DCYACS001073",  # Disconnected Youth , 16-24 not working or in school
        "S1101SPHOUSE006073",  # single parent household
    ]

    df_a, df_m = pull_data_from_geodb(geodb, series_ids)

    df = df_a.append(Wrangler.annualize(df_m))
    df = Wrangler.long_to_wide(df)
    df = Wrangler.drop_years(df, start=2010, end=2018)
    df = Wrangler.replace_nan(df, median=
        ['Gross Domestic Product: All Industries', 'SNAP Benefits Recipients']
    )
    df = Wrangler.normalize_features(df)

    print("Exporting clean data...")
    my_path = "/home/adam/github/Intern-Proj-1/features-v1.csv"
    df.to_csv(my_path, index=False)
    print(f"\nData available at {my_path}\n")
    breakpoint()

    return df


if __name__ == "__main__":
    main()
