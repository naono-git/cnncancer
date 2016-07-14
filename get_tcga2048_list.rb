#!/usr/bin/env ruby

srcDirList=["/Users/nono/Documents/data/tissue_images/TCGA-05-4384-01A-01-BS1_2048_files/15",
            "/Users/nono/Documents/data/tissue_images/TCGA-38-4631-11A-01-TS1_2048_files/14",
            "/Users/nono/Documents/data/tissue_images/TCGA-05-4425-01A-01-BS1_2048_files/15"]

fout = File.open("filelist.txt","w")

3.times do |aa|
##srcDirList.each do |srcDir|
  srcDir = srcDirList[aa]
  srcList = Dir.glob("#{srcDir}/*_*.jpeg")
  srcList.each do |ss|
    if(File.size(ss) > 2e+5) then
      fout.print("#{ss}\t#{aa}\n")
    end
  end
end
