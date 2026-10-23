using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

static class FrameIO
{
    public static Frame Load(string path)
    {
        using var bmp = new Bitmap(path);
        var d = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
        var a = new int[bmp.Width * bmp.Height];
        Marshal.Copy(d.Scan0, a, 0, a.Length);
        bmp.UnlockBits(d);
        return Frame.From(a, bmp.Width, bmp.Height);
    }

    public static Frame LoadCrop(string path, int x, int y, int w, int h)
    {
        using var full = new Bitmap(path);
        using var bmp = full.Clone(new Rectangle(x, y, Math.Min(w, full.Width - x), Math.Min(h, full.Height - y)), PixelFormat.Format32bppArgb);
        var d = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
        var a = new int[bmp.Width * bmp.Height];
        Marshal.Copy(d.Scan0, a, 0, a.Length);
        bmp.UnlockBits(d);
        return Frame.From(a, bmp.Width, bmp.Height);
    }
}
