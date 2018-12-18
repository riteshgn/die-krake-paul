import sqlite3
import pandas as pd


class KaggleExtractor(object):

    def __init__(self):
        self._conn = None

    def __enter__(self):
        self._conn = sqlite3.connect('soccer/database.sqlite')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._conn.close()

    def prepare_dataset(self, season, league_id):
        print('Starting extraction... [season={season}, league: {league}]'.format(season=season, league=league_id))
        df_home_player_stats = self._collect_home_player_stats(season, league_id)
        df_away_player_stats = self._collect_away_player_stats(season, league_id)
        df_player_avg_overall_ratings = self._get_player_overall_attribute_rating()

        df_home_player_stats = _enhance_player_stats(df_home_player_stats, 'hp', df_player_avg_overall_ratings)
        # print(df_home_player_stats[['hp1_age', 'hp1_avg_overall_rating', 'hp1_height', 'hp1_weight']].head())
        # print('\n')

        df_away_player_stats = _enhance_player_stats(df_away_player_stats, 'ap', df_player_avg_overall_ratings)
        # print(df_away_player_stats[['ap1_age', 'ap1_avg_overall_rating', 'ap1_height', 'ap1_weight']].head())
        # print('\n')

        df_matches = self._collect_match_stats(season, league_id)
        df_match_results = df_matches[['id', 'result']]

        df_matchday_player_stats = df_home_player_stats.merge(df_away_player_stats, on='id')
        df_matchday_player_stats = df_matchday_player_stats.merge(df_match_results, on='id')

        columns_to_drop = ['hp{index}_id'.format(index=plyr_indx) for plyr_indx in range(1, 12)]
        columns_to_drop = columns_to_drop + ['hp{index}_bday'.format(index=plyr_indx) for plyr_indx in range(1, 12)]
        columns_to_drop = columns_to_drop + ['ap{index}_id'.format(index=plyr_indx) for plyr_indx in range(1, 12)]
        columns_to_drop = columns_to_drop + ['ap{index}_bday'.format(index=plyr_indx) for plyr_indx in range(1, 12)]
        columns_to_drop = columns_to_drop + ['id', 'date_x']
        final_data_set = df_matchday_player_stats.drop(columns=columns_to_drop)
        # print(final_data_set.columns)
        # print('\n')

        filename = 'kaggle_db/features_{}_{}_v1.csv'.format(season.replace('/', '_'), league_id)
        final_data_set.to_csv(filename)
        print('Extraction complete... [(matches, feature_count): {shape} saved to file: {filename}]'.format(
            records=final_data_set.size, shape=final_data_set.shape, filename=filename))

    def _collect_home_player_stats(self, season, league_id):
        query = \
            'select ' \
            '   m.id, m.date, ' \
            '   m.home_player_1 as hp1_id, p1.birthday as hp1_bday, p1.height as hp1_height, p1.weight as hp1_weight, ' \
            '   m.home_player_2 as hp2_id, p2.birthday as hp2_bday, p2.height as hp2_height, p2.weight as hp2_weight, ' \
            '   m.home_player_3 as hp3_id, p3.birthday as hp3_bday, p3.height as hp3_height, p3.weight as hp3_weight, ' \
            '   m.home_player_4 as hp4_id, p4.birthday as hp4_bday, p4.height as hp4_height, p4.weight as hp4_weight, ' \
            '   m.home_player_5 as hp5_id, p5.birthday as hp5_bday, p5.height as hp5_height, p5.weight as hp5_weight, ' \
            '   m.home_player_6 as hp6_id, p6.birthday as hp6_bday, p6.height as hp6_height, p6.weight as hp6_weight, ' \
            '   m.home_player_7 as hp7_id, p7.birthday as hp7_bday, p7.height as hp7_height, p7.weight as hp7_weight, ' \
            '   m.home_player_8 as hp8_id, p8.birthday as hp8_bday, p8.height as hp8_height, p8.weight as hp8_weight, ' \
            '   m.home_player_9 as hp9_id, p9.birthday as hp9_bday, p9.height as hp9_height, p9.weight as hp9_weight, ' \
            '   m.home_player_10 as hp10_id, p10.birthday as hp10_bday, p10.height as hp10_height, p10.weight as hp10_weight, ' \
            '   m.home_player_11 as hp11_id, p11.birthday as hp11_bday, p11.height as hp11_height, p11.weight as hp11_weight ' \
            'from Match m ' \
            '   join League l on l.id=m.league_id ' \
            '   join Player p1 on p1.player_api_id=m.home_player_1 ' \
            '   join Player p2 on p2.player_api_id=m.home_player_2 ' \
            '   join Player p3 on p3.player_api_id=m.home_player_3 ' \
            '   join Player p4 on p4.player_api_id=m.home_player_4 ' \
            '   join Player p5 on p5.player_api_id=m.home_player_5 ' \
            '   join Player p6 on p6.player_api_id=m.home_player_6 ' \
            '   join Player p7 on p7.player_api_id=m.home_player_7 ' \
            '   join Player p8 on p8.player_api_id=m.home_player_8 ' \
            '   join Player p9 on p9.player_api_id=m.home_player_9 ' \
            '   join Player p10 on p10.player_api_id=m.home_player_10 ' \
            '   join Player p11 on p11.player_api_id=m.home_player_11 ' \
            'where season="{season}" and l.id={league_id} ' \
            'order by m.date asc'.format(season=season, league_id=league_id)
        df_home_player_stats = pd.read_sql_query(query, self._conn)
        features = ['hp1_bday', 'hp1_height', 'hp1_weight', 'hp11_bday', 'hp11_height', 'hp11_weight']
        # print(df_home_player_stats[features].head())
        # print('\n')
        return df_home_player_stats

    def _collect_away_player_stats(self, season, league_id):
        query = \
            'select ' \
            '   m.id, m.date, ' \
            '   m.away_player_1 as ap1_id, p1.birthday as ap1_bday, p1.height as ap1_height, p1.weight as ap1_weight, ' \
            '   m.away_player_2 as ap2_id, p2.birthday as ap2_bday, p2.height as ap2_height, p2.weight as ap2_weight, ' \
            '   m.away_player_3 as ap3_id, p3.birthday as ap3_bday, p3.height as ap3_height, p3.weight as ap3_weight, ' \
            '   m.away_player_4 as ap4_id, p4.birthday as ap4_bday, p4.height as ap4_height, p4.weight as ap4_weight, ' \
            '   m.away_player_5 as ap5_id, p5.birthday as ap5_bday, p5.height as ap5_height, p5.weight as ap5_weight, ' \
            '   m.away_player_6 as ap6_id, p6.birthday as ap6_bday, p6.height as ap6_height, p6.weight as ap6_weight, ' \
            '   m.away_player_7 as ap7_id, p7.birthday as ap7_bday, p7.height as ap7_height, p7.weight as ap7_weight, ' \
            '   m.away_player_8 as ap8_id, p8.birthday as ap8_bday, p8.height as ap8_height, p8.weight as ap8_weight, ' \
            '   m.away_player_9 as ap9_id, p9.birthday as ap9_bday, p9.height as ap9_height, p9.weight as ap9_weight, ' \
            '   m.away_player_10 as ap10_id, p10.birthday as ap10_bday, p10.height as ap10_height, p10.weight as ap10_weight, ' \
            '   m.away_player_11 as ap11_id, p11.birthday as ap11_bday, p11.height as ap11_height, p11.weight as ap11_weight ' \
            'from Match m ' \
            '   join League l on l.id=m.league_id ' \
            '   join Player p1 on p1.player_api_id=m.away_player_1 ' \
            '   join Player p2 on p2.player_api_id=m.away_player_2 ' \
            '   join Player p3 on p3.player_api_id=m.away_player_3 ' \
            '   join Player p4 on p4.player_api_id=m.away_player_4 ' \
            '   join Player p5 on p5.player_api_id=m.away_player_5 ' \
            '   join Player p6 on p6.player_api_id=m.away_player_6 ' \
            '   join Player p7 on p7.player_api_id=m.away_player_7 ' \
            '   join Player p8 on p8.player_api_id=m.away_player_8 ' \
            '   join Player p9 on p9.player_api_id=m.away_player_9 ' \
            '   join Player p10 on p10.player_api_id=m.away_player_10 ' \
            '   join Player p11 on p11.player_api_id=m.away_player_11 ' \
            'where season="{season}" and l.id={league_id} ' \
            'order by m.date asc'.format(season=season, league_id=league_id)
        df_away_player_stats = pd.read_sql_query(query, self._conn)
        features = ['ap1_bday', 'ap1_height', 'ap1_weight', 'ap11_bday', 'ap11_height', 'ap11_weight']
        # print(df_away_player_stats[features].head())
        # print('\n')
        return df_away_player_stats

    def _get_player_overall_attribute_rating(self):
        query = 'select pattr.player_api_id, avg(pattr.overall_rating) as avg_overall_rating ' \
                'from Player_Attributes pattr group by pattr.player_api_id'
        df_player_avg_overall_ratings = pd.read_sql_query(query, self._conn)
        # print(df_player_avg_overall_ratings.head())
        # print('\n')
        return df_player_avg_overall_ratings

    def _collect_match_stats(self, season, league_id):
        query = 'select ' \
                '   m.id, c.name as country, l.name as league, m.stage, m.date, ' \
                '   t1.team_short_name as home_team, t2.team_short_name as away_team, ' \
                '   m.home_team_goal as home_goals, m.away_team_goal as away_goals ' \
                'from Match m ' \
                '   join Country c on c.id=m.country_id ' \
                '   join League l on l.id=m.league_id ' \
                '   join Team t1 on t1.team_api_id=m.home_team_api_id ' \
                '   join Team t2 on t2.team_api_id=m.away_team_api_id ' \
                'where season="{season}" and l.id={league_id} ' \
                'order by m.date asc'.format(season=season, league_id=league_id)
        df_matches = pd.read_sql_query(query, self._conn)
        features = ['stage', 'home_team', 'home_goals', 'away_goals', 'away_team']
        # print(df_matches[features].head())
        # print('\n')

        df_matches = df_matches.apply(_set_result, axis=1)
        return df_matches


def _append_player_ages(player, player_type, df_player_avg_overall_ratings):
    for plyr_indx in range(1, 12):
        key_plyr_age = '{prefix}{index}_age'.format(prefix=player_type, index=plyr_indx)
        key_plyr_bday = '{prefix}{index}_bday'.format(prefix=player_type, index=plyr_indx)
        player[key_plyr_age] = (player['date'] - player[key_plyr_bday]).days / 365

        key_plyr_id = '{prefix}{index}_id'.format(prefix=player_type, index=plyr_indx)
        key_plyr_avg_overall_rating = '{prefix}{index}_avg_overall_rating'.format(prefix=player_type, index=plyr_indx)

        plyr_ratings = df_player_avg_overall_ratings[
            df_player_avg_overall_ratings['player_api_id'] == player[key_plyr_id]]
        player[key_plyr_avg_overall_rating] = plyr_ratings['avg_overall_rating'].iloc[0]
    return player


def _enhance_player_stats(df, player_type, df_player_avg_overall_ratings):
    df['date'] = df['date'].apply(pd.to_datetime)
    for plyr_indx in range(1, 12):
        df['{prefix}{index}_bday'.format(prefix=player_type, index=plyr_indx)] = \
            df['{prefix}{index}_bday'.format(prefix=player_type, index=plyr_indx)].apply(pd.to_datetime)

    return df.apply(lambda x: _append_player_ages(x, player_type, df_player_avg_overall_ratings), axis=1)


def _set_result(match):
    if match['home_goals'] == match['away_goals']:
        match['result'] = 'D'
    else:
        match['result'] = 'H' if match['home_goals'] > match['away_goals'] else 'A'
    return match


def main():
    source_matches = [
        {'season': '2008/2009', 'league_id': 1729},
        {'season': '2009/2010', 'league_id': 1729},
        {'season': '2010/2011', 'league_id': 1729},
        {'season': '2011/2012', 'league_id': 1729},
        {'season': '2012/2013', 'league_id': 1729},
        {'season': '2013/2014', 'league_id': 1729},
        {'season': '2014/2015', 'league_id': 1729},
        {'season': '2015/2016', 'league_id': 1729},
    ]

    extractor = KaggleExtractor()
    with extractor:
        for source in source_matches:
            extractor.prepare_dataset(source['season'], source['league_id'])


if __name__ == '__main__':
    main()
