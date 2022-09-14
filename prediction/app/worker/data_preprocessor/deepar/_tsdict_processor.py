import pandas as pd
from loguru import logger

"""NOTE: Commented code not be removed, this is for basic version to get working"""


class TsDictProcessor:
    """Preprocessing pandas dataframe into train, valid and test"""

    def __init__(self, data, validate_end_dtm, frequency, min_num_zero=6):
        """preprocess data into a time series dict
        data(pandas.DataFrame): time series data to  be converted to json
        """
        self.fill = 0
        # self.levels = data_column_list
        self.min_num_nonzeros = min_num_zero
        self._data = data
        self.end_date = validate_end_dtm
        self.freq = frequency

        if isinstance(self._data, pd.DataFrame):
            self._data = self.data_clean()
        else:
            raise Exception
            logger.exception("Not a pandas data frame")

        # running preliminary checks and set level variables
        # self.preliminary_checks()

        # run following functions
        # self.group_data()
        self.make_ts_dict()
        # self.ts_filtered_out()

    @property
    def data(self):
        return self._data

    @property
    def ts_dict(self):
        return self._ts_dict

    # def preliminary_checks(self):
    #     """Checking for required column names in dataframe
    #     """
    #     for level in self.levels:
    #         if level not in self.data.columns:
    #             # checking for levels in data frame
    #             raise logger.exception(f'Column "{level}" not found')

    #     if self.date_col not in self.data.columns:
    #         raise logger.exception(f'Column\t:"{self.date_col}" not found')

    # def data_clean(self):
    #     """Basic cleaning of dataframe
    #     """
    #     logger.info(f'Remove records where column <{self.value_col}> has negative/zero values')
    #     self._data = self._data.loc[self._data[self.value_col] > 0, :]
    #     for level in self.levels:
    #         self._data.loc[:, level] = self._data.loc[:, level].apply(str)
    #     return self._data

    def data_clean(self):
        self._data = self._data.clip(lower=0)
        return self._data

    # def group_data(self):
    #     group = self.levels + [pd.Grouper(key=self.date_col, freq=self.freq)]
    #     self.grouped_data = self.data.groupby(group)[self.value_col].sum()
    #     self.uniq_grps = {grp[:-1] for grp in self.grouped_data.index}

    # def ts_checks(self, df, group):
    #     """ This is a function to check whether a timeseries meet desired conditions.
    #     Time series have to pass all checks to be in the
    #     processed timeseries dictionary. You may add new check_functions
    #     in this module and call the same within this function. Make sure that
    #     function you define should return `True` to pass check.
    #     """
    #     # check inactivity from the given date, minimum non zeros, minlength of df
    #     checks = [check_min_nonzeros(df, self.value_col, self.min_num_nonzeros),
    #               dormancy_check(df, self.value_col)]

    #     return all(checks)

    def make_ts_dict(self):
        self._ts_dict = {}
        for col in self.data.columns:
            target_df = reindex_series(
                self.data[col], self.end_date, self.freq, fill=self.fill
            )
            self._ts_dict[col] = target_df
        # pickle_to_s3(self.ts_dict, self.bucket, f'{self.out_prefix}/ts_dict.pkl')

    # def make_ts_dict(self):
    #     self._ts_dict = {}
    #     for grp in self.uniq_grps:
    #         df = reindex_series(self.grouped_data.loc[grp], self.end_date,
    #                             self.freq, fill=self.fill)

    #         if self.ts_checks(df, grp):
    #             self._ts_dict[grp] = df
    #             logger.debug(f'Group : {grp}')
    #     logger.info(f'Total # of Groups : {len(self.uniq_grps)}')
    #     logger.info(f'Processed # of Groups : {len(self.ts_dict.keys())}')

    # def ts_filtered_out(self):
    #     self.filtered = set(self.uniq_grps) - set(self.ts_dict.keys())
    #     logger.info(f'Filtered # of Groups : {len(self.filtered)}')


################### time series processing functions ##################


def reindex_series(df, endate, freq, fill=0):
    idx = pd.date_range(start=df.index.min(), end=endate, freq=freq)
    return df.reindex(index=idx, fill_value=fill)


##################### time series check functions #####################

# def check_min_nonzeros(df, value_col, min_num_nonzeros):
#     """ check if the number of non zero entries in
#     dataframe[value_col]> `min_num_nonzeros`
#     """
#     return sum(df > 0) > min_num_nonzeros

# def dormancy_check(df, value_col, n_months=7):
#     """ check will pass the if the grouped df has nonzero values in last `n_months`
#     """
#     return sum(df.iloc[-n_months:] > 0) >= 1
