require "json"

str = ""
open("url.json", "r:utf-8") do |f|
  f.each_line { |line| str << line }
end

root = JSON.parse(str)

def find_url(src, dst)
  if src.kind_of?(Array)
    src.each { |item| find_url(item, dst) }
  elsif src.kind_of?(String)
    if src.start_with?("http") && !src.include?("encrypted-tbn0.gstatic.com")
      dst << src
    end
  end
end

urls = []
find_url(root, urls)
open("url.txt", "w:utf-8") do |f|
  urls.each { |url| f.puts(url) }
end
