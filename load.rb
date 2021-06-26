require "net/http"
require "openssl"
require "json"
require "base64"
require "pathname"

def get_image_ext(data)
  case data
  when /GIF8[79]a/n
    return "gif"
  when /\x89PNG/n
    return "png"
  when /\xFF\xD8/n
    return "jpg"
  else
    raise
  end
end

# How to use?
# (1) get instance like: " a = AutoLoader.new({url_list},{thread_size},[{des_dir},{ca_file}])"
# (2) let it run!  : " a.load() "
# (3) its progress will be shown and more details writen into "log.txt"
class AutoLoader

  #
  #@param src URLリストのファイル
  #@param size threadの数
  #
  def initialize(src, dst, thread_size = 4)
    @dst = Pathname.new(dst)
    @queue = []
    File.open(src) do |file|
      file.each_line do |line|
        @queue.push(line.chomp!)
      end
    end
    @num = thread_size
    @queue_mutex = Mutex.new
    @log_mutex = Mutex.new
    @result_mutex = Mutex.new
  end

  def get(uri)
    begin
      return Net::HTTP.get_response(uri)
    rescue => e
      log e.class
      log e.message
    end
    return nil
  end

  def load()
    threads = []
    @log = open("log.txt", "w")
    @success = 0
    @cnt = 0
    @size = @queue.length
    print "start..."
    for i in 0..@num
      thread = Thread.new do
        while e = dequeue
          r = process(e)
          on_processed(r)
        end
      end
      threads.push(thread)
    end
    threads.each { |t| t.join }
    log("All done.")
    @log.close
    puts "\nAll done."
  end

  def process(path)
    uri = URI.parse(path)
    path = Pathname.new(uri.path)
    name = @dst + path.basename
    return false if File.exists?(name)
    index = 1
    while File.exist?(name)
      name = @dst + "#{path.basename(".*")}_#{index}#{path.extname}"
      index += 1
    end
    response = get(uri)
    if response == nil
      log "Error > #{path}"
    elsif response.code == "200"
      img = response.body
      if ![".png", ".jpg", ".gif"].include?(name.extname)
        name = name.sub_ext("." + get_image_ext(img))
      end
      open(name, "wb") { |f| f.write(img) }
      log "Success > #{uri}"
      return true
    else
      log "Error > #{response.code} #{uri}"
    end
    return false
  end

  def on_processed(result)
    @result_mutex.synchronize do
      @cnt += 1
      if result then @success += 1 end
      print "\r  #{(100.0 * @cnt / @size).to_i}%  success:#{@success}/#{@cnt}"
    end
  end

  def dequeue()
    @queue_mutex.synchronize { return @queue.shift }
  end

  def log(mes)
    @log_mutex.synchronize { @log.puts(mes) }
  end
end

AutoLoader.new(ARGV[0], ARGV[1]).load()
