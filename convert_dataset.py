import os
import pandas as pd
import numpy as np
import awkward
from sklearn.model_selection import train_test_split

def load_data(dark_path, qcd_path, path):

  dark_path = dark_path
  qcd_path = qcd_path
  
  Darkdata = pd.read_csv(dark_path) 
  QCDdata = pd.read_csv(qcd_path)
  Darkdata.columns = ['pt', 'DeltaEta', 'DeltaPhi', 'DeltaR', 'xpT']
  QCDdata.columns = ['pt', 'DeltaEta', 'DeltaPhi', 'DeltaR', 'xpT']

  #convert csv file to h5
  Darkdata = pd.read_csv(dark_path) 
  QCDdata = pd.read_csv(qcd_path)

  Darkdata.columns = ['pt', 'DeltaEta', 'DeltaPhi', 'DeltaR', 'xpT']
  QCDdata.columns = ['pt', 'DeltaEta', 'DeltaPhi', 'DeltaR', 'xpT']

  Darkdata = pd.DataFrame.to_numpy(Darkdata)
  QCDdata = pd.DataFrame.to_numpy(QCDdata)


  # convert tracks files to h5 file
  columns = []
  for i in range(50):
    columns.append('PT_' + str(i))
    columns.append('DeltaEta_' + str(i))
    columns.append('DeltaPhi_' + str(i))
    columns.append('DeltaR_' + str(i))
    columns.append('xpT_' + str(i))

  df_darktrack = pd.DataFrame(columns=columns)
  jets = int(len(Darkdata)/50)
  for i in range(jets):
    tracks  = i*50
    row = pd.Series(np.concatenate(Darkdata[tracks:tracks + 50]), index=columns).transpose()
    row['label'] = 0
    df_darktrack = df_darktrack.append(row, ignore_index=True)

  df_darktrack.to_hdf(path + 'darktrack.h5', key='df_darktrack', mode='w')
  print('created Dark h5 file')

  df_qcdtrack = pd.DataFrame(columns=columns)
  jets = int(len(QCDdata)/50)
  for i in range(jets):
    tracks  = i*50
    row = pd.Series(np.concatenate(QCDdata[tracks:tracks + 50]), index=columns).transpose()
    row['label'] = 1
    df_qcdtrack = df_qcdtrack.append(row, ignore_index=True)
  df_qcdtrack.to_hdf(path + 'qcdtrack.h5', key='df_qcdtrack', mode='w')
  print('created QCD h5 file')

  # load h5 files
  df_darktrack = pd.read_hdf(path + 'darktrack.h5')
  df_qcdtrack = pd.read_hdf(path + 'qcdtrack.h5')


  # split data to train, validation and test
  frac_train = int(0.75*df_qcdtrack.shape[0]*0.9)
  val_frac = int(0.75*df_qcdtrack.shape[0]*0.1)

  train = pd.concat([df_darktrack[:frac_train], df_qcdtrack[:frac_train]])
  val = pd.concat([df_darktrack[frac_train:frac_train + val_frac], df_qcdtrack[frac_train:frac_train + val_frac]])
  qcd_test = pd.concat([df_qcdtrack[frac_train + val_frac:]])
  dark_test = pd.concat([df_darktrack[frac_train + val_frac:]])

  import logging
  logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')

  def _transform(dataframe, start=0, stop=-1, jet_size=0.8):
    from collections import OrderedDict
    v = OrderedDict()

    df = dataframe.iloc[start:stop]
    def _col_list(prefix, max_particles=50):
      return ['%s_%d'%(prefix,i) for i in range(max_particles)]

    _PT = df[_col_list('PT')].values
    _DeltaEtaTrack = df[_col_list('DeltaEta')].values
    _DeltaPhiTrack = df[_col_list('DeltaPhi')].values
    _DeltaRTrack = df[_col_list('DeltaR')].values
    _xpT = df[_col_list('xpT')].values

    mask = _PT>0
    n_particles = np.sum(mask, axis=1)

    pt = awkward.JaggedArray.fromcounts(n_particles, _PT[mask])
    eta = awkward.JaggedArray.fromcounts(n_particles, _DeltaEtaTrack[mask])
    phi = awkward.JaggedArray.fromcounts(n_particles, _DeltaPhiTrack[mask])
    r = awkward.JaggedArray.fromcounts(n_particles, _DeltaRTrack[mask])
    xpt = awkward.JaggedArray.fromcounts(n_particles, _xpT[mask])

    #outputs
    _label = df['label'].values
    v['label'] = np.stack((_label, 1-_label), axis=-1)

    v['n_parts'] = n_particles
    v['part_pt_log'] = np.log(pt)
    v['part_ptrel'] = xpt
    v['part_logptrel'] = np.log(v['part_ptrel'])
    v['part_etarel'] = eta
    v['part_phirel'] = phi
    v['part_deltaR'] = r

    return v

  def convert(source, destdir, basename, step=None, limit=None):
      df = source
      logging.info('Total events: %s' % str(df.shape[0]))
      if limit is not None:
          df = df.iloc[0:limit]
          logging.info('Restricting to the first %s events:' % str(df.shape[0]))
      if step is None:
          step = df.shape[0]
      idx=-1
      while True:
          idx+=1
          start=idx*step
          if start>=df.shape[0]: break
          if not os.path.exists(destdir):
              os.makedirs(destdir)
          output = os.path.join(destdir, '%s_%d.awkd'%(basename, idx))
          logging.info(output)
          if os.path.exists(output):
              logging.warning('... file already exist: continue ...')
              continue
          v=_transform(df, start=start, stop=start+step)
          awkward.save(output, v, mode='x')


  convert(train, destdir=path, basename='trackTrain')
  convert(val, destdir=path, basename='trackValidation')
  convert(qcd_test, destdir=path, basename='qcd_test')
  convert(dark_test, destdir=path, basename='dark_test')
  