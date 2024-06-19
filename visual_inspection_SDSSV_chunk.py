#!/usr/local/anaconda3/bin/python

import os,sys,time
import matplotlib
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from astropy import units as u
from astropy import constants as const
import pandas as pd
from scipy.signal import medfilt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, LogLocator,
                               AutoMinorLocator, ScalarFormatter)
warnings.filterwarnings("ignore")

def read_sdss5_spec(filename):
    spec = fits.getdata(filename, 1)
    wave = 10**spec.LOGLAM
    flux = spec.FLUX
    error = np.zeros(len(spec.IVAR))
    error= 1/np.sqrt(spec.IVAR)
    good_ind = np.where((np.isfinite(flux))&(np.isfinite(error)))[0]
    return wave[good_ind], flux[good_ind], error[good_ind]

def plot_spec(fig, wave, flux, error, z, info, smooth=1):
    # plot rest-frame spectrum
    wave = wave/(1+z)
    flux = flux*(1+z)
    error = error*(1+z)
    
    if smooth > 1:
        flux = medfilt(flux, smooth)
        error = medfilt(error, smooth)
    medflux = medfilt(flux, 5*smooth)

    nw = 0
    for wd in range(len(window_xrange)):
        if (wave.min()<window_xrange[wd][0]) and (wave.max()>window_xrange[wd][1]):
            nw += 1
    #fig, ax = plt.subplots(2, nw, figsize=(12, 8))
    axes = fig.add_subplot(2,1,1)
    axes.set_position([0.08, 0.52, 0.88, 0.4])

    axes.plot(wave, flux, c='k', ls='-', lw=0.8)
    axes.plot(wave, medfilt(flux, 15*smooth), c='seagreen', ls='-', lw=1)
    axes.plot(wave, error, c='r', ls='-', lw=0.8, alpha=.7)

    for ll in range(len(line_cen)):
        if (line_cen[ll] > wave.min()) and (line_cen[ll] < wave.max()):
            axes.axvline(line_cen[ll], ls='--', c='gray', lw=0.5)
            axes.text(line_cen[ll], 1.25*medflux[100:-50].max(), line_name[ll], fontsize=10, rotation=90, va='top')
    if z < 0.1: axes.set_title(info, fontsize=18, loc='left', c='r')
    else: axes.set_title(info, fontsize=18, loc='left', c='k')
    axes.set_xlim(wave.min(), wave.max())
    axes.set_ylim(medflux[100:-50].min()-0.2*abs(medflux[100:-50].min()), 1.3*abs(medflux[100:-50].max()))
    axes.xaxis.set_major_locator(MultipleLocator(500.))
    axes.xaxis.set_minor_locator(MultipleLocator(50))
    axes.set_xlabel(r'Rest-frame Wavelength ($\rm\AA$)', fontsize=16)
    axes.text(-0.06, -0.05, r'$ f_{\lambda}$ ($\rm 10^{-17} {\rm erg\;s^{-1}\;cm^{-2}\;\AA^{-1}}$)', fontsize=16,
        transform=axes.transAxes, rotation=90, ha='center', rotation_mode='anchor')
    #axes.set_ylabel(r'$ f_{\lambda}$ ($\rm 10^{-17} {\rm erg\;s^{-1}\;cm^{-2}\;\AA^{-1}}$)', fontsize=18)

    nwi = 0
    dfig = (0.88-(nw-1)*0.04)/nw
    for wd in range(len(window_xrange)):
        if (wave.min()<window_xrange[wd][0]) and (wave.max()>window_xrange[wd][1]):
            if nw > 1: axis = fig.add_subplot(2, nw, 2)
            else: axis = fig.add_subplot(2, nw, nw+nwi)
            axis.set_position([0.08+nwi*(dfig+0.04), 0.05, dfig, 0.36])
            axis.set_title(window_name[wd], fontsize=15)
            ind_wd = np.where((wave>window_xrange[wd][0])&(wave<window_xrange[wd][1]))[0]
            axis.plot(wave[ind_wd], flux[ind_wd], c='k', ls='-', lw=0.8, alpha=1)
            axis.plot(wave[ind_wd], error[ind_wd], c='r', ls='-', lw=0.8, alpha=0.7)
            for ll in range(len(line_cen)):
                if (line_cen[ll] > window_xrange[wd][0]) and (line_cen[ll] < window_xrange[wd][1]):
                    axis.axvline(line_cen[ll], ls='--', c='gray', lw=0.8)           
            axis.set_xlim(window_xrange[wd])     
            nwi += 1
    return fig

def plot_spec2(fig, wave, flux, error, z, info, smooth=1):
    # plot observed-frame spectrum
    if smooth > 1:
        flux = medfilt(flux, smooth)
        error = medfilt(error, smooth)
    medflux = medfilt(flux, 5*smooth)

    nw = 0
    for wd in range(len(window_xrange)):
        if (wave.min()<window_xrange[wd][0]*(1+z)) and (wave.max()>window_xrange[wd][1]*(1+z)):
            nw += 1
    #fig, ax = plt.subplots(2, nw, figsize=(12, 8))
    #axes = plt.subplot(2,1,1)
    axes = fig.add_subplot(2,1,1)
    axes.set_position([0.08, 0.52, 0.88, 0.4])

    axes.plot(wave, flux, c='k', ls='-', lw=0.8)
    axes.plot(wave, medfilt(flux, 15*smooth), c='seagreen', ls='-', lw=1)
    axes.plot(wave, error, c='r', ls='-', lw=0.8, alpha=.7)
    
    for ll in range(len(line_cen)):
        if (line_cen[ll]*(1+z) > wave.min()) and (line_cen[ll]*(1+z) < wave.max()):
            axes.axvline(line_cen[ll]*(1+z), ls='--', c='gray', lw=0.5)
            axes.text(line_cen[ll]*(1+z), 1.25*medflux[100:-50].max(), line_name[ll], fontsize=10, rotation=90, va='top')
    if wave.min() < 5890 < wave.max():
        axes.axvline(5890, ls='--', c='r', lw=0.5)
        axes.text(5890, 1.25*medflux[100:-50].max(), 'NaD', fontsize=10, rotation=90, va='top', color='r')
    if z < 0.1: axes.set_title(info, fontsize=18, loc='left', c='r')
    else: axes.set_title(info, fontsize=18, loc='left', c='k')
    axes.set_xlim(wave.min(), wave.max())
    axes.set_ylim(medflux[100:-50].min()-0.2*abs(medflux[100:-50].min()), 1.3*abs(medflux[100:-50].max()))
    axes.xaxis.set_major_locator(MultipleLocator(500.))
    axes.xaxis.set_minor_locator(MultipleLocator(50))
    axes.set_xlabel(r'Obs-frame Wavelength ($\rm\AA$)', fontsize=16)
    axes.text(-0.06, -0.05, r'$ f_{\lambda}$ ($\rm 10^{-17} {\rm erg\;s^{-1}\;cm^{-2}\;\AA^{-1}}$)', fontsize=16,
        transform=axes.transAxes, rotation=90, ha='center', rotation_mode='anchor')
    #axes.set_ylabel(r'$ f_{\lambda}$ ($\rm 10^{-17} {\rm erg\;s^{-1}\;cm^{-2}\;\AA^{-1}}$)', fontsize=18)

    nwi = 0
    dfig = (0.88-(nw-1)*0.04)/nw
    for wd in range(len(window_xrange)):
        if (wave.min()<window_xrange[wd][0]*(1+z)) and (wave.max()>window_xrange[wd][1]*(1+z)):
            if nw > 1: axis = fig.add_subplot(2, nw, 2)
            else: axis = fig.add_subplot(2, nw, nw+nwi)
            axis.set_position([0.08+nwi*(dfig+0.04), 0.05, dfig, 0.36])
            axis.set_title(window_name[wd], fontsize=15)
            ind_wd = np.where((wave>window_xrange[wd][0]*(1+z))&(wave<window_xrange[wd][1]*(1+z)))[0]
            axis.plot(wave[ind_wd], flux[ind_wd], c='k', ls='-', lw=0.8, alpha=1)
            axis.plot(wave[ind_wd], error[ind_wd], c='r', ls='-', lw=0.8, alpha=0.7)
            for ll in range(len(line_cen)):
                if (line_cen[ll] > window_xrange[wd][0]) and (line_cen[ll] < window_xrange[wd][1]):
                    axis.axvline(line_cen[ll]*(1+z), ls='--', c='gray', lw=0.8)           
            axis.set_xlim(window_xrange[wd]*(1+z))     
            nwi += 1
    return fig


if __name__ == "__main__":
    line_cen = np.array([6564.61,  6732.66, 5875, 4862.68, 5008.24, 4687.02, 4341.68, 3934.78, 3728.47, \
                     3426.84, 2798.75, 1908.72, 1816.97, 1750.26, 1718.55, 1549.06, 1640.42, 1402.06, 1396.76, 1335.30, \
                     1215.67])
    line_name = np.array(['Ha+[NII]','[SII]6718,6732', 'HeI', 'Hb', '[OIII]', 'HeII4687', 'Hr', 'CaII3934', '[OII]3728', \
                        'NeV3426', 'MgII', 'CIII]', 'SiII1816', 'NIII]1750', 'NIV]1718', 'CIV', 'HeII1640', '', 'SiIV+OIV', \
                        'CII1335', 'Lya'])
    window_xrange = np.array([[1150, 1290], [1500, 1700], [1700, 1970], \
                            [2700, 2900], [4640, 5100], [6400, 6800],])
    window_name = np.array(['Lya', 'CIV', 'CIII]', 'MgII', 'Hbeta', 'Halpha'])

    # Read the input catalog
    input_catalog = './qso_vi_medfid_15000-19999.fits'
    print('\nReading catalog from the following file:\n'+input_catalog+'\n')
    print('Press N to update the catalog path, or press any other key to continue.')
    response = input()
    if (response == 'N') or (response == 'n'):
        input_catalog = input('Enter the new catalog path: ')
        print('Reading catalog from the following file:\n'+input_catalog+'\n')
        print('Press N to update the catalog path, or press any other key to continue.')
        response = input()

    if os.path.exists(input_catalog):
        vi_catalog = fits.getdata(input_catalog, 1)

        print('Number of QSOs in the catalog: '+str(len(vi_catalog))+'\n')
        
        # Start visual inspection
        print('Starting visual inspection.')
        print('Press the index where you would like to start, or press any other key to start from the beginning.')
        response = input('Input the index: ')
        if response.isdigit():
            if 0 < int(response) < len(vi_catalog):
                start_index = int(response)
            else:
                start_index = 0
        else:
            start_index = 0

        print('Will start from index: '+str(start_index))
        
        # Loop over the QSOs
        dump_results = 'INDEX,FIELD,MJD,CATALOGID,Z,RA,DEC,COMMENT,\n'
        print("1. Press Enter to skip the object \n2. Enter 'exit' to exit visual inspection \n"\
              +"3. Enter 'obs' to switch to observed-frame spectrum (default is rest-frame)\n"\
              +"4. Enter 'rf' to switch to rest-frame spectrum\n"\
              +"5. Enter 'z=' to update the redshift \n"\
              +"6. Enter 'smooth=' to update the smoothing factor \n"\
              +"7. Enter 'nonqso' to mark the object as non-QSO \n"\
              +"8. Enter 'mark:' to mark the object for further inspection.")
        for ind_obj in range(start_index, len(vi_catalog)):
            obj_line = ''
            field_obj, mjd_obj, catalogID_obj, sdssID_obj = vi_catalog.FIELD[ind_obj], vi_catalog.MJD[ind_obj], vi_catalog.CATALOGID[ind_obj], vi_catalog.SDSS_ID[ind_obj]
            ra_obj, dec_obj, z_obj = vi_catalog.RACAT[ind_obj], vi_catalog.DECCAT[ind_obj], vi_catalog.Z[ind_obj]
            specfile_obj = vi_catalog.SPEC_FILE[ind_obj]
            class_obj, firstcarton_obj = vi_catalog.CLASS[ind_obj], vi_catalog.FIRSTCARTON[ind_obj]
            if vi_catalog.FIELD[ind_obj]<16046:
                spec_file_path = '/data2/sdss5/v6_0_4/%05dp/coadd/%05d/spS'%(field_obj, mjd_obj)+specfile_obj[1:]
            else:
                spec_file_path = '/data4/sdss5/master/spectra/lite/%06d/%05d/'%(field_obj, mjd_obj)+specfile_obj
            if os.path.exists(spec_file_path):
                wave_obj, flux_obj, error_obj = read_sdss5_spec(spec_file_path)
                info_obj = '%06d-%-5d-%d      z=%.4f\nCLASS=%s     FIRSTCARTON=%s'%(field_obj, mjd_obj, catalogID_obj, z_obj, class_obj, firstcarton_obj)    
                if ind_obj == start_index:
                    fig = plt.figure(figsize=(12, 8))
                fig = plot_spec(fig, wave_obj, flux_obj, error_obj, z_obj, info_obj, smooth=1)
                plt.draw()
                plt.show(block=False)
                time.sleep(0.1)
                response = input()
                if response == 'exit':
                    break
                flag = True
                pix_smooth = 1
                z_new = z_obj
                while flag:
                    if response == '':
                        flag = False
                        if z_new != z_obj:
                            obj_line = str(ind_obj)+','+str(field_obj)+','+str(mjd_obj)+','+str(catalogID_obj)+','+str(z_obj)+','+str(ra_obj)+','+str(dec_obj)+',z=%f,\n'%(z_new)
                        else:
                            obj_line = str(ind_obj)+','+str(field_obj)+','+str(mjd_obj)+','+str(catalogID_obj)+','+str(z_obj)+','+str(ra_obj)+','+str(dec_obj)+',,\n'
                    elif response == 'nonqso':
                        flag = False
                        obj_line = str(ind_obj)+','+str(field_obj)+','+str(mjd_obj)+','+str(catalogID_obj)+','+str(z_obj)+','+str(ra_obj)+','+str(dec_obj)+',nonqso,\n'
                    elif response[:5] == 'mark:':
                        flag = False
                        obj_line = str(ind_obj)+','+str(field_obj)+','+str(mjd_obj)+','+str(catalogID_obj)+','+str(z_obj)+','+str(ra_obj)+','+str(dec_obj)+','+response[5:]+',\n'
                    elif response == 'obs':
                        fig.clf()
                        fig = plot_spec2(fig, wave_obj, flux_obj, error_obj, z_new, info_obj, smooth=pix_smooth)
                        plt.draw()
                        plt.show(block=False)
                        time.sleep(0.1)
                        response = input()
                    elif response == 'rf':
                        fig.clf()
                        fig = plot_spec(fig, wave_obj, flux_obj, error_obj, z_new, info_obj, smooth=pix_smooth)
                        plt.draw()
                        plt.show(block=False)
                        time.sleep(0.1)
                        response = input()
                    elif response[:7] == 'smooth=':
                        pix_smooth = int(response[7:])
                        fig.clf()
                        fig = plot_spec(fig, wave_obj, flux_obj, error_obj, z_new, info_obj, smooth=pix_smooth)
                        plt.draw()
                        plt.show(block=False)
                        time.sleep(0.1)
                        response = input()
                    elif response[:2] == 'z=':
                        z_new = float(response[2:])
                        info_obj = '%06d-%-5d-%d      z=%.4f\nCLASS=%s     FIRSTCARTON=%s'%(field_obj, mjd_obj, catalogID_obj, z_new, class_obj, firstcarton_obj)    
                        fig.clf()
                        fig = plot_spec(fig, wave_obj, flux_obj, error_obj, z_new, info_obj, smooth=pix_smooth)
                        plt.draw()
                        plt.show(block=False)
                        time.sleep(0.1)
                        response = input()
                    elif response == 'exit':
                        break                        
                    else:
                        print('Invalid input. Please try again.')
                        print("1. Press Enter to skip the object \n2. Enter 'exit' to exit visual inspection \n"\
                                +"3. Enter 'obs' to switch to observed-frame spectrum (default is rest-frame)\n"\
                                +"4. Enter 'rf' to switch to rest-frame spectrum\n"\
                                +"5. Enter 'z=' to update the redshift \n"\
                                +"6. Enter 'smooth=' to update the smoothing factor \n"\
                                +"7. Enter 'nonqso' to mark the object as non-QSO \n"\
                                +"8. Enter 'mark:' to mark the object for further inspection.")
                        response = input()
                if response == 'exit':
                    break
                fig.clf()
                print('Object index '+str(ind_obj)+' done.')
                dump_results += obj_line
            else:
                print('The spectrum file does not exist. Skipping the object.')
        
        # Save the results
        with open('vi_'+datetime.today().strftime('%Y-%m-%d_%H:%M:%S')+'.dat', 'w') as file:
            file.write(dump_results)
        print('Exit. Results saved in the file: vi_'+datetime.today().strftime('%Y-%m-%d_%H:%M:%S')+'.dat')
    else:
        print('The input catalog does not exist. Exiting the program.')