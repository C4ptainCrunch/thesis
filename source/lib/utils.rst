  
Utilitiy functions
------------------




  


  .. code:: ipython3

    ENABLE_JOB_SUBMISSION = False
    
    try:
        import boto3
        client = boto3.client('batch')
    except:
        ENABLE_JOB_SUBMISSION = False
        print("AWS is not configured")
    
    
    def submit_aws_job(*args, local=False, **kwargs):
        if ENABLE_JOB_SUBMISSION and not local:
            return client.submit_job(*args, **kwargs)






  


  .. code:: ipython3

    import random
    
    def max_rand(iterable, key=lambda x: x):
        '''Get the greatest element of an iterable and solve ties by a coin flip'''
        maximum_value = max(key(x) for x in iterable)
        keep = [x for x in iterable if key(x) == maximum_value]
        return random.choice(keep)






  


  .. code:: ipython3

    import numpy as np
    
    def fold(matrix):
        folded = (matrix + (1 - np.transpose(matrix))) / 2
    
        for i in range(folded.shape[0]):
            for j in range(folded.shape[1]):
                if i > j:
                    folded[i, j] = None
        return folded




