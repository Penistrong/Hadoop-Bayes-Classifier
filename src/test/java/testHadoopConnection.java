import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;

public class testHadoopConnection {
    Configuration configuration;
    FileSystem fileSystem;

    @Before
    public void conn() throws IOException {
        //1.配置
        configuration = new Configuration(true);
        //设置Hadoop用户名，不然是使用Windows的用户名，没有相关读写权限
        System.setProperty("HADOOP_USER_NAME", "hadoop");
        //2.文件系统
        fileSystem = FileSystem.get(configuration);
    }

    @After
    public void close() throws IOException {
        fileSystem.close();
    }

    @Test
    public void mkdir() throws IOException {
        Path path = new Path("/helloHadoopFromIDEA");
        if (fileSystem.exists(path)) {
            fileSystem.delete(path, true);
        }
        fileSystem.mkdirs(path);
    }

}
