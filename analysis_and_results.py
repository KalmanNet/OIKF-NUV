import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np

def plot_MSE(r_squared_vec_dB, P_MSE_dB, P_sigma_dB,scl,p):

  plt.figure(figsize=(12, 8))
  plt.plot(r_squared_vec_dB, P_MSE_dB[0,:], '--', color='tab:red', linewidth=3 ,label='Noise floor')
  plt.plot(r_squared_vec_dB, P_MSE_dB[1,:], '--', color='tab:green', linewidth=3 ,label='KF')
  plt.plot(r_squared_vec_dB, P_MSE_dB[2,:], '-s', linewidth=2 ,color='tab:blue',markersize=15 , label='WRKF')
  plt.plot(r_squared_vec_dB, P_MSE_dB[5,:], '-^',linewidth=2 , color='tab:orange',markersize=18, label=r'$\chi^2$ - Test')
  plt.plot(r_squared_vec_dB, P_MSE_dB[3, :], '-p', color='tab:cyan',linewidth=2.5 ,markersize=19,  label='NUV-AM')
  plt.plot(r_squared_vec_dB, P_MSE_dB[4,:], '-o',color='tab:pink',linewidth=2.5 ,markersize=15 ,label='NUV-EM')
  plt.xticks(fontsize=26)
  plt.yticks(fontsize=26)
  plt.ylabel('MSE [dB]', size='36')
  plt.xlabel(r"$r^{-2}$ [dB]", size='40')
  plt.grid()
  plt.legend(fontsize='18',loc='lower left')

  plt.tight_layout()
  plt.savefig('results/KF_noisy.pdf', bbox_inches='tight')
  plt.show()

  plt.figure(figsize=(12, 8))
  plt.plot(r_squared_vec_dB, P_MSE_dB[6,:], '--', color='tab:red', linewidth=3 ,label='Noise floor')
  plt.plot(r_squared_vec_dB, P_MSE_dB[7,:], '--', color='tab:green', linewidth=3 ,label='KF')
  plt.plot(r_squared_vec_dB, P_MSE_dB[8,:], '-s', linewidth=2 ,color='tab:blue',markersize=16 , label='WRKF')
  plt.plot(r_squared_vec_dB, P_MSE_dB[11,:], '-^',linewidth=2 , color='tab:orange',markersize=18, label=r'$\chi^2$ - Test')
  plt.plot(r_squared_vec_dB, P_MSE_dB[9, :], '-p', color='tab:cyan',linewidth=2.5 ,markersize=19,  label='NUV-AM')
  plt.plot(r_squared_vec_dB, P_MSE_dB[10,:], '-o',color='tab:pink',linewidth=2.5 ,markersize=16 ,label='NUV-EM')
  plt.xticks(fontsize=26)
  plt.yticks(fontsize=26)
  plt.ylabel('MSE [dB]', size='36')
  plt.xlabel(r"$r^{-2}$ [dB]", size='36')
  plt.grid()
  plt.legend(fontsize='18',loc='lower left')

 
  plt.tight_layout()
  plt.savefig('results/KF_noisy_with_outliers_p='+str(p)+'_scl='+str(scl)+'.pdf', bbox_inches='tight')
  plt.show()


###############################################################################

  for i in range(2):

    def int2str(MSE,sigam):
      return str(np.round(MSE,2))+u"\n\u00B1"+str(np.round(sigam,2))

    tab = ['results/KF_tab_noisy','results/KF_tab_noisy_with_outliers']
    j = 0
    if i == 1:
      j = i + 5
    table = [[u"1/r\u00B2 [dB]" ,                 '-10',                                 '0',                               '10',                                 '20',                               '30'],
            [u'Noise floor [\u0302\u03BC\u00B1\u0302\u03C3]', int2str(P_MSE_dB[0+j,0],P_sigma_dB[0+j,0]),   int2str(P_MSE_dB[0+j,1],P_sigma_dB[0+j,1]),   int2str(P_MSE_dB[0+j,2], P_sigma_dB[0+j,2]),   int2str(P_MSE_dB[0+j,3],P_sigma_dB[0+j,3]),  int2str(P_MSE_dB[0+j,4], P_sigma_dB[0+j,4])],
            [u'KF [\u0302\u03BC\u00B1\u0302\u03C3]',          int2str(P_MSE_dB[1+j,0],P_sigma_dB[1+j,0]),   int2str(P_MSE_dB[1+j,1],P_sigma_dB[1+j,1]),   int2str(P_MSE_dB[1+j,2], P_sigma_dB[1+j,2]),   int2str(P_MSE_dB[1+j,3],P_sigma_dB[1+j,3]),  int2str(P_MSE_dB[1+j,4], P_sigma_dB[1+j,4])],
            [u'WRKF [\u0302\u03BC\u00B1\u0302\u03C3]',        int2str(P_MSE_dB[2+j,0],P_sigma_dB[2+j,0]),   int2str(P_MSE_dB[2+j,1],P_sigma_dB[2+j,1]),   int2str(P_MSE_dB[2+j,2], P_sigma_dB[2+j,2]),   int2str(P_MSE_dB[2+j,3],P_sigma_dB[2+j,3]),  int2str(P_MSE_dB[2+j,4], P_sigma_dB[2+j,4])],
            [u'Chi-squared [\u0302\u03BC\u00B1\u0302\u03C3]', int2str(P_MSE_dB[5+j,0],P_sigma_dB[5+j,0]),   int2str(P_MSE_dB[5+j,1],P_sigma_dB[5+j,1]),   int2str(P_MSE_dB[5+j,2], P_sigma_dB[5+j,2]),   int2str(P_MSE_dB[5+j,3],P_sigma_dB[5+j,3]),  int2str(P_MSE_dB[5+j,4], P_sigma_dB[5+j,4])],
            [u'Joint MAP (AM)[\u0302\u03BC\u00B1\u0302\u03C3]',     int2str(P_MSE_dB[3+j,0],P_sigma_dB[3+j,0]),   int2str(P_MSE_dB[3+j,1],P_sigma_dB[3+j,1]),   int2str(P_MSE_dB[3+j,2], P_sigma_dB[3+j,2]),   int2str(P_MSE_dB[3+j,3],P_sigma_dB[3+j,3]),  int2str(P_MSE_dB[3+j,4], P_sigma_dB[3+j,4])],
            [u'OIKF-EM [\u0302\u03BC\u00B1\u0302\u03C3]',     int2str(P_MSE_dB[4+j,0],P_sigma_dB[4+j,0]),   int2str(P_MSE_dB[4+j,1],P_sigma_dB[4+j,1]),   int2str(P_MSE_dB[4+j,2], P_sigma_dB[4+j,2]),   int2str(P_MSE_dB[4+j,3],P_sigma_dB[4+j,3]),  int2str(P_MSE_dB[4+j,4], P_sigma_dB[4+j,4])]]

    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
    with open(tab[i]+'_p='+str(p)+'_scl='+str(scl)+'.txt', 'w', encoding="utf-8") as f:
      f.write(tabulate(table, headers='firstrow', tablefmt='latex'))
    f.close()