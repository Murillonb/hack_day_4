import pickle

# limpeza, transformações e encodes
class CostaDelData(object):
    def __init__(self):
        self.moda                               = pickle.load(open('parameter/moda.pkl', 'rb'))
        self.rs_meses_da_reserva_ate_o_check_in = pickle.load(open('parameter/rs_meses_da_reserva_ate_o_check_in.pkl', 'rb'))
        self.rs_numero_de_pernoites_reservadas  = pickle.load(open('parameter/rs_numero_de_pernoites_reservadas.pkl', 'rb'))
        self.rs_numero_de_hospedes              = pickle.load(open('parameter/rs_numero_de_hospedes.pkl', 'rb'))
        self.nacio_dict                         = pickle.load(open('parameter/nacio_dict.pkl', 'rb'))
        self.le_forma_de_reserva                = pickle.load(open('parameter/le_forma_de_reserva.pkl', 'rb'))
        self.le_tipo_do_quarto_reservado        = pickle.load(open('parameter/le_tipo_do_quarto_reservado.pkl', 'rb'))

    
    def data_cleaning(self, df1):      
        ## 1.1. Verificando as Colunas

        cols_old = ['id', 'Classificacao do hotel', 'Meses da reserva ate o check-in',
                    'Numero de pernoites reservadas', 'Numero de hospedes',
                    'Regime de alimentacao', 'Nacionalidade', 'Forma de Reserva',
                    'Ja se hospedou anteriormente', 'Tipo do quarto reservado',
                    'Reserva feita por agencia de turismo', 'Reserva feita por empresa',
                    'Reserva com Estacionamento', 'Reserva com Observacoes']

        snakecase = lambda x: x.replace(' ', '_').lower()

        cols_new = list(map(snakecase, cols_old))

        df1.columns = cols_new

        ## 1.5. Preenchendo os NA

        # numero_de_hospedes - substituiremos pela moda
        df1.loc[df1['numero_de_hospedes'].isna(), 'numero_de_hospedes'] = self.moda
    
        ## 1.6. Alterando os tipos dos dados

        df1.loc[:, 'numero_de_hospedes'] = df1['numero_de_hospedes'].astype('int64')
           
        return df1
    
    def data_preparation(self, df5):
        # 5.0. PREPARAÇÃO DOS DADOS

        ## 5.2.Redimensionamento

        # meses_da_reserva_ate_o_check-in
        df5['meses_da_reserva_ate_o_check-in'] = self.rs_meses_da_reserva_ate_o_check_in.transform(df5[['meses_da_reserva_ate_o_check-in']].values)

        # numero_de_pernoites_reservadas
        df5['numero_de_pernoites_reservadas'] = self.rs_numero_de_pernoites_reservadas.transform(df5[['numero_de_pernoites_reservadas']].values)

        # numero_de_hospedes
        df5['numero_de_hospedes'] = self.rs_numero_de_hospedes.transform(df5[['numero_de_hospedes']].values)

        ## 5.3. Encoding

        # classificacao_do_hotel - Ordinal Encoding
        class_hotel_dict = {'4 estrelas': 1, '5 estrelas': 2}
        df5['classificacao_do_hotel'] = df5['classificacao_do_hotel'].map(class_hotel_dict)

        # regime_de_alimentacao - Ordinal Encoding
        reg_alim_dict = {'Sem refeicao': 0, 
                         'Café da manha': 1,
                         'Café da manha e jantar': 2, 
                         'Café da manha, almoco e jantar': 3}
        df5['regime_de_alimentacao'] = df5['regime_de_alimentacao'].map(reg_alim_dict)

        # nacionalidade - Target Encoding
        df5['nacionalidade'] = df5['nacionalidade'].map(self.nacio_dict)

        # forma_de_reserva - Label Encoding
        df5['forma_de_reserva'] = self.le_forma_de_reserva.transform(df5['forma_de_reserva'])

        # tipo_do_quarto_reservado - Label Encoding
        df5['tipo_do_quarto_reservado'] = self.le_tipo_do_quarto_reservado.transform(df5['tipo_do_quarto_reservado'])

        # reserva_com_observacoes - Ordinal Encoding
        res_obs_dict = {'Nenhuma': 0,
                        '1 a 3': 1, 
                        'Mais de 3': 2}
        df5['reserva_com_observacoes'] = df5['reserva_com_observacoes'].map(res_obs_dict)

        # dicionário comum
        dict_yes_no = {'Sim': 1, 'Não': 0}

        # ja_se_hospedou_anteriormente - Ordinal Encoding
        df5['ja_se_hospedou_anteriormente'] = df5['ja_se_hospedou_anteriormente'].map(dict_yes_no)

        # reserva_feita_por_agencia_de_turismo - Ordinal Encoding
        df5['reserva_feita_por_agencia_de_turismo'] = df5['reserva_feita_por_agencia_de_turismo'].map(dict_yes_no)

        # reserva_feita_por_empresa - Ordinal Encoding
        df5['reserva_feita_por_empresa'] = df5['reserva_feita_por_empresa'].map(dict_yes_no)

        # reserva_com_estacionamento - Ordinal Encoding
        df5['reserva_com_estacionamento'] = df5['reserva_com_estacionamento'].map(dict_yes_no)
        
        # selecionando colunas
        cols_selected = ['id', 'classificacao_do_hotel', 'meses_da_reserva_ate_o_check-in', 
                         'numero_de_pernoites_reservadas', 'nacionalidade', 'reserva_com_estacionamento', 
                         'reserva_com_observacoes']
        
        df5 = df5.fillna(0)        
        
        return df5[cols_selected]
        
    def get_prediction(self,  model, original_data, test_data):
        # preditor
        pred = model.predict( test_data )
                
        original_data['predictions'] = pred
        
        return original_data.to_json(orient='records')