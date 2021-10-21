
def fetch_dataset_name(idx):
    name_number_dict = {18: 'EighteenSrcsRdm', 
                            14:'FourteenSrcsRdm', 
                            13:'ThirteenSrcsRdm',
                            19:'NineteenSrcsRdm', 
                            12:'TwelveSrcsRdm', 
                            17:'SeventeenSrcsRdm',
                            20:'TwentySrcsRdm', 
                            15:'FifteenSrcsRdm', 
                            16:'SixteenSrcsRdm',
                           11:'ElevenSrcsRdm',
                           1:'1SourcesRdm',
                           2:'2SourcesRdm',
                           3:'3SourcesRdm',
                           4:'4SourcesRdm',
                           5:'5SourcesRdm',
                           6:'6SourcesRdm',
                           7:'7SourcesRdm',
                       8:'8SourcesRdm',
                       9:'9SourcesRdm',
                       10:'10SourcesRdm'}
    return name_number_dict[idx]
    
