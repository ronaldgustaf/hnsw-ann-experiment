results = {'sift':  {'brute': {'build_time': 0.5026280879974365, 
                                'query_time': 279.39087867736816},
                        'hnsw': {'recall': 0.70828,
                                'build_time': 123.6417624950409,
                                'query_time': 0.14356136322021484},
                        'flann': {'recall': 0.82904,
                                'build_time': 638.7001960277557,
                                'query_time': 13.044065952301025},
                        'annoy': {'recall': 0.9267200000000001,
                                'build_time': 273.279093503952,
                                'query_time': 22.34026789665222}},
        'glove': {'brute': {'build_time': 0.6018815040588379,
                              'query_time': 338.47492504119873},
                        'hnsw': {'recall': 0.4706,
                                'build_time': 246.98338413238525,
                                'query_time': 0.2142331600189209},
                        'flann': {'recall': 0.9999899999999999,
                                'build_time': 1691.0208961963654,
                                'query_time': 456.82367396354675},
                        'annoy': {'recall': 0.7164299999999999,
                                'build_time': 367.3854081630707,
                                'query_time': 37.131975412368774}},
        'deep': {'brute': {'build_time': 0.5273756980895996,
                              'query_time': 260.1447868347168},
                    'hnsw': {'recall': 0.7000700000000001,
                              'build_time': 130.78997421264648,
                              'query_time': 0.1455221176147461},
                      'flann': {'recall': 0.9093900000000001,
                                'build_time': 633.8056845664978,
                                'query_time': 19.652721643447876},
                      'annoy': {'recall': 0.94752,
                                'build_time': 307.4670329093933,
                                'query_time': 37.44424319267273}},
          'mnist': {'brute': {'build_time': 0.05738949775695801,
                            'query_time': 134.8580060005188},
                    'hnsw': {'recall': 0.92871,
                            'build_time': 17.066320657730103,
                            'query_time': 0.40122151374816895},
                    'flann': {'recall': 0.8596,
                            'build_time': 179.88845205307007,
                            'query_time': 4.117311716079712},
                    'annoy': {'recall': 0.99325,
                            'build_time': 49.07278060913086,
                            'query_time': 33.98891043663025}},
                                }

hnsw_tuned_results = {'sift': {'hnsw_tuned': {'recall': 0.8235299999999999,
                                                'build_time': 490.23624992370605,
                                                'query_time': 0.29738807678222656}},
                        'glove': {'hnsw_tuned': {'recall': 0.66635,
                                                'build_time': 1511.0596585273743,
                                                'query_time': 0.5713100433349609}},
                        'deep': {'hnsw_tuned': {'recall': 0.8231200000000001,
                                                'build_time': 440.2416741847992,
                                                'query_time': 0.2549624443054199}},
                        'mnist': {'hnsw_tuned': {'recall': 0.9471800000000001,
                                                'build_time': 31.10383129119873,
                                                'query_time': 0.41745567321777344}}}