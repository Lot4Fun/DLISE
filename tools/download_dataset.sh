#!/bin/bash

# Config: General
save_dir="/archive/DLISE/download"/$(date "+%Y%m%d")
# Config: CMEMS
cmems_ssh_url="ftp://nrt.cmems-du.eu/Core/GLOBAL_ANALYSISFORECAST_PHY_CPL_001_015/MetO-GLO-PHY-CPL-dm-SSH"
cmems_sst_url="ftp://nrt.cmems-du.eu/Core/GLOBAL_ANALYSISFORECAST_PHY_CPL_001_015/MetO-GLO-PHY-CPL-dm-TEM"
cmems_bio_url="ftp://nrt.cmems-du.eu/Core/GLOBAL_ANALYSIS_FORECAST_BIO_001_028/global-analysis-forecast-bio-001-028-daily"
years='
  2018
  2019
  2020
  2021
'
# Config: Argo
argo_urls='
  https://ocg.aori.u-tokyo.ac.jp/member/eoka/data/NPargodata/data/2018.lzh
  https://ocg.aori.u-tokyo.ac.jp/member/eoka/data/NPargodata/data/2019.lzh
  https://ocg.aori.u-tokyo.ac.jp/member/eoka/data/NPargodata/data/2020Jan.lzh
  https://ocg.aori.u-tokyo.ac.jp/member/eoka/data/NPargodata/data/2020Feb.lzh
  https://ocg.aori.u-tokyo.ac.jp/member/eoka/data/NPargodata/data/2020Mar.lzh
  https://ocg.aori.u-tokyo.ac.jp/member/eoka/data/NPargodata/data/2020Apr.lzh
  https://ocg.aori.u-tokyo.ac.jp/member/eoka/data/NPargodata/data/2020May.lzh
  https://ocg.aori.u-tokyo.ac.jp/member/eoka/data/NPargodata/data/2020Jun.lzh
  https://ocg.aori.u-tokyo.ac.jp/member/eoka/data/NPargodata/data/2020Jul.lzh
  https://ocg.aori.u-tokyo.ac.jp/member/eoka/data/NPargodata/data/2020Aug.lzh
  https://ocg.aori.u-tokyo.ac.jp/member/eoka/data/NPargodata/data/2020Sep.lzh
  https://ocg.aori.u-tokyo.ac.jp/member/eoka/data/NPargodata/data/2020Oct.lzh
  https://ocg.aori.u-tokyo.ac.jp/member/eoka/data/NPargodata/data/2020Nov.lzh
'

# Ask login ID and password
read -p "CMEMS ID: " cmems_id
read -sp "CMEMS Password: " cmems_pass
tty -s && echo
echo ${cmems_id} ${cmems_pass}

# Download SSH and SST
types=("ssh" "sst" "bio")
for type in ${types}; do

  tmp_save_dir=${save_dir}/${type}
  mkdir -p ${tmp_save_dir}
  cd ${tmp_save_dir}

  for year in ${years}; do
    
    if [ ${type} = "ssh" ]; then
      download_url=${cmems_ssh_url}/${year}
    elif  [ ${type} = "sst" ]; then
      download_url=${cmems_sst_url}/${year}
    else
      download_url=${cmems_bio_url}/${year}
    fi

    wget -r -nd --user=${cmems_id} --password=${cmems_pass} ${download_url}

  done

done

# Download Argo
tmp_save_dir=${save_dir}/argo
mkdir -p ${tmp_save_dir}
cd ${tmp_save_dir}

for argo_url in ${argo_urls}; do
  wget ${argo_url}
done

for file in `\find ${tmp_save_dir} -maxdepth 1 -type f`; do
  echo ${file}
  lha x ${file}
done

rm ${tmp_save_dir}/*.lzh

touch ${save_dir}/Download_finished
echo "Download has finished"
