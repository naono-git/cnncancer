#!/usr/bin/env ruby

# srcDirList=[["TCGA-05-4384-01A-01-BS1_2048_files/15",
#              "/project/hikaku_db/data/tissue_images/TCGA-05-4405-01A-01-BS1_2048_files/15",
#              "/project/hikaku_db/data/tissue_images/TCGA-05-4410-01A-02-BS2_2048_files/15"],
#             ["/project/hikaku_db/data/tissue_images/TCGA-38-4631-11A-01-TS1_2048_files/14",
#              "/project/hikaku_db/data/tissue_images/TCGA_05-4390-01A-01-BS1_2048_files/15",
#              "/project/hikaku_db/data/tissue_images/TCGA-35-3615-01A-01-BS1_2048_files/16"],
#             ["/project/hikaku_db/data/tissue_images/TCGA-05-4425-01A-01-BS1_2048_files/15",
#              "/project/hikaku_db/data/tissue_images/TCGA-44-6775-01A-01-TS3_2048_files/16",
#              "/project/hikaku_db/data/tissue_images/TCGA-44-7662-01A-01-BS1_2048_files/15"]]

srcDir0 = "/project/hikaku_db/data/tissue_images"
tmp = Dir.glob(srcDir0+"/TCGA-*2048_files")
srcDirList = tmp.map do |dd|
  tmp2 = Dir.glob(dd+"/*")
  tmp3 = tmp2.map{|ff| File.basename(ff).to_i}.sort
  "#{dd}/#{tmp3.pop}"
end

subtypeTable = File.open("/project/hikaku_db/data/tissue_images/tcga.luad.gene.expression.subtypes.20121025.csv").readlines
subtypeTable.shift
subtype = subtypeTable.map do |line|
  line.chomp.split(",")
end

subtypeList = srcDirList.map do |dd|
  hoge = subtype.find do |st|
    dd.include?(st[0])
  end
  hoge[1]
end

File.open("subtypeList.txt","w") do |fout|
  srcDirList.each_with_index do |dd,aa|
    fout.print("#{dd.gsub("/project/hikaku_db/data/tissue_images/","")}\t#{subtypeList[aa]}\n")
  end
end

File.open("./sizeList.txt", "w") do |fout|
  srcDirList.each_with_index do |dd,aa|
    p dd
    srcList = Dir.glob("#{dd}/*_*.jpeg")
    srcList.each do |ss|
      fout.print(ss.gsub("/project/hikaku_db/data/tissue_images/",""),"\t",
                 File.size(ss),"\t",
                 `identify -format \"%[fx:w] %[fx:h]\" #{ss}`)
    end
  end
end

fout = File.open("filelist_2048.txt","w")

srcDirList.each_with_index do |srcDir,aa|
  tt = subtypeList[aa]
  srcList = Dir.glob("#{srcDir}/*_*.jpeg")
  fuga = srcList.map do |ss|
    xy = IO.popen("identify -format \"%[fx:w] %[fx:h]\" #{ss}",'r').read().chomp
  end
  srcList.each_with_index do |ss,aa|
    xy=fuga[aa]
    if(xy!="2048 2048") then
      next
    end
    if(File.size(ss) < 2e+5) then
      next 
    end
    fout.print("#{ss}\t#{tt}\n")
  end
end
