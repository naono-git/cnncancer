#!/usr/bin/env ruby

srcDirList=[["/project/hikaku_db/data/tissue_images/TCGA-05-4384-01A-01-BS1_2048_files/15",
             "/project/hikaku_db/data/tissue_images/TCGA-05-4405-01A-01-BS1_2048_files/15",
             "/project/hikaku_db/data/tissue_images/TCGA-05-4410-01A-02-BS2_2048_files/15"],
            ["/project/hikaku_db/data/tissue_images/TCGA-38-4631-11A-01-TS1_2048_files/14",
             "/project/hikaku_db/data/tissue_images/TCGA_05-4390-01A-01-BS1_2048_files/15",
             "/project/hikaku_db/data/tissue_images/TCGA-35-3615-01A-01-BS1_2048_files/16"],
            ["/project/hikaku_db/data/tissue_images/TCGA-05-4425-01A-01-BS1_2048_files/15",
             "/project/hikaku_db/data/tissue_images/TCGA-44-6775-01A-01-TS3_2048_files/16",
             "/project/hikaku_db/data/tissue_images/TCGA-44-7662-01A-01-BS1_2048_files/15"]]

p IO.popen("identify -format \"%[fx:w] %[fx:h]\" /project/hikaku_db/data/tissue_images/TCGA-05-4390-01A-02-BS2_2048_files/15/0_0.jpeg").read()

fout = File.open("filelist.txt","w")

3.times do |aa|
  3.times do |bb|
    ##srcDirList.each do |srcDir|
    srcDir = srcDirList[aa][bb]
    srcList = Dir.glob("#{srcDir}/*_*.jpeg")
    srcList.each do |ss|
      hoge = IO.popen("identify -format \"%[fx:w] %[fx:h]\" #{ss}",'r').read().chomp
      if(hoge!="2048 2048") then
        next
      end
      if(File.size(ss) < 2e+5) then
        next 
      end
      fout.print("#{ss}\t#{aa}\n")
    end
  end
end
