  
Download data from AWS
=======================




  


  .. code:: ipython3

    %matplotlib inline
    
    import boto3
    import time
    import json
    import os
    
    import numpy as np
    import pandas as pd
    import math
    
    from datetime import datetime, timezone
    from datetime import timedelta as td
    
    session = boto3.client('logs')













  


  .. code:: ipython3

    def download_results():
        start = int(datetime(2021, 5, 1).timestamp())
        results = []
    
        while True:
            print('Querying')
    
            # Query for the first time beginning in 1970
            query = session.start_query(
                logGroupName='/aws/batch/job',
                startTime=start,
                endTime=int(time.time()),
                queryString="""
                    fields @message, @timestamp
                    | filter success == 1
                    | sort @timestamp asc
                    | limit 10000
                """,
                limit=10000
            )
    
            # Loop untill the query response is ready
            query_result = {'status': 'Scheduled'}
            while query_result['status'] in ('Scheduled', 'Running'):
                query_result = session.get_query_results(queryId=query['queryId'])
                time.sleep(1)
    
            tmp = query_result
    
            # If we reached the end, we don't have any more data, we break
            if len(query_result['results']) == 0:
                break
    
            # If we did not reach the end, we get the timestamp of the last line and use this
            # as a starting point for the next iteration
            assert query_result['results'][-1][1]['field'] == "@timestamp"
            start = int(datetime.strptime(query_result['results'][-1][1]['value'], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()) + 1
    
            # Process the data a little bit and push it to results
            data = [
                json.loads(row[0]['value'])
                for row in query_result['results']
            ]
            results += data
    
        print("Done")
        
        return results






  


  .. code:: ipython3

    SOURCE_DIR = os.path.dirname(os.path.dirname(__file__))
    
    
    DATA_DIR = os.path.join(SOURCE_DIR, "data/")
    CLOUDWATCH_DATA_PATH = os.path.join(DATA_DIR, "cloudwtach-results.jsonl")






  


  .. code:: ipython3

    def write_results(results):
        with open(CLOUDWATCH_DATA_PATH, "w") as fd:
            for row in results:
                json.dump(row, fd)
                fd.write("\n")






  


  .. code:: ipython3

    #write_results(download_results())






.. parsed-literal::

    Querying
    Querying
    Done







  


  .. code:: ipython3

    import glob
    
    def read_results():
        with open(CLOUDWATCH_DATA_PATH) as fd:
            cloudwatch = [json.loads(l) for l in fd]
    
            local = []
    
            for path in glob.glob(os.path.join(DATA_DIR, "local-results", "*.jsonl")):
                with open(path) as fd:
                    data = [json.loads(l) for l in fd if l.startswith("{")]
                local += data
    
        raw_results = cloudwatch + local
    
        for row in raw_results:
            if row['winner'] is None:
                row['winner'] = 0.5
    
        df = pd.DataFrame(raw_results)
        return df






  


  .. code:: ipython3

    def get_locals():
        class Player:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
    
        class GreedyPlayer(Player):
            pass
    
        class MCTSPlayer(Player):
            pass
    
        class UCTPlayer(Player):
            pass
    
        class GreedyUCTPlayer(Player):
            pass
    
        class RandomPlayer(Player):
            pass
        
        class AlphaBetaMinimaxPlayer(Player):
            pass
        
        return locals()






  


  .. code:: ipython3

    player_locals = get_locals()






  


  .. code:: ipython3

    def process_results(df):
        
    
        # Fix bad data
        df['version'] = df.version.fillna(1).astype(int)
        df['side'] = df['side'].fillna(-1).astype(int)
    
        # Only keep usable data
        df = df[df.version >= 2]
    
        is_normalized = df['side'] == 0
    
        to_normalise = df[~is_normalized].copy()
    
        # Swap columns
        opponents = to_normalise['opponent'].copy()
        players = to_normalise['player'].copy()
    
        to_normalise['player'] = opponents
        to_normalise['opponent'] = players
    
        # Swap winner
        to_normalise['winner'] = 1 - to_normalise['winner']
    
        # Swap individual scores
        to_normalise['score'] = to_normalise['score'].map(lambda x: list(reversed(x)))
    
        # Concatenate the 2 sides
        normalised = pd.concat([
            df[is_normalized].copy(),
            to_normalise.copy()
        ]).copy()
    
    
        # Evaluate both players
        normalised['player_eval'] = normalised['player'].map(lambda x: eval(x, globals(), player_locals))
        normalised['opponent_eval'] = normalised['opponent'].map(lambda x: eval(x, globals(), player_locals))
        
        return normalised






  


  .. code:: ipython3

    raw_results = read_results()






  


  .. code:: ipython3

    results = process_results(raw_results)




