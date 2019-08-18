# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

# [0.4.0] - 2019-02-07
- Update sphinx docs
- Fix discretized p-value bug
- Fix zero p-value bug
- Add MGC/Dcorr Time Series
- Raised some package requirements to fix errors with called packages

## [0.3.0] - 2019-02-07
- Refactor the mgcpy package structure
- Add Manova
- Fix two sample tests
- Add scripts for k sample test power curves
- Update sphinx docs

## [0.2.0] - 2019-01-25
- Add FastDCorr
- Update FastMGC to latest and more stable code
- Update FastMGC and FastDCorr unit test
- Update MGC's Smoothing and Distance Transform functions
- Fix broken unit tests

## [0.1.2] - 2019-01-17
- Fix setup.py to be able to build from source when wheel is not found
- Add MANIFEST.in to include Cython `*.pyx` files

## [0.1.1] - 2019-01-12
- Port repo from `NeuroDataDesign` to `neurodata` org
- Assert dims of input matrices

## [0.1.0] - 2018-12-14
### Added
- Port [MGC](https://github.com/neurodata/mgc)/[FastMGC](https://github.com/neurodata/mgc-matlab) into the package, by [@tpsatish95](https://github.com/tpsatish95)
- Port HHG, Pearson/RV/Cca, Spearman/Kendall into package and add data simulations, by [@sampan501](https://github.com/sampan501)
- Port [dcorr/mcorr/mantel](https://github.com/neurodata/mgc-matlab), power estimation, and validate implementations, by [@junhaobearxiong](https://github.com/junhaobearxiong)
- Port [MDMR](https://github.com/FCP-INDI/C-PAC/blob/master/CPAC/cwas/mdmr.pyx) into package by, [@sundaysundya](https://github.com/sundaysundya)
- Implement Random Forest Independence Test by, [@rguo123](https://github.com/rguo123) [not in `master` yet]
- Implement 2-sample tests into package by [@ananyas713](https://github.com/ananyas713)
