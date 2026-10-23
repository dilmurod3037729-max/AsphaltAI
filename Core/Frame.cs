/// <summary>Rang kanallarining minimumi (oq piksel = yuqori). Raqamlar oq, fon rangli/qorong'i.</summary>
public sealed class Frame
{
    public int W, H;
    public byte[] White;

    public static Frame From(int[] argb, int w, int h)
    {
        var m = new byte[argb.Length];
        for (int i = 0; i < m.Length; i++)
        {
            int c = argb[i], r = c >> 16 & 255, g = c >> 8 & 255, b = c & 255;
            m[i] = (byte)Math.Min(r, Math.Min(g, b));
        }
        return new Frame { W = w, H = h, White = m };
    }

    /// <summary>RGBA_8888 bayt buferidan (qator qadami bilan) faqat qirqimni oladi.</summary>
    public static Frame FromRgba(ReadOnlySpan<byte> buf, int stride, int x0, int y0, int w, int h)
    {
        var m = new byte[w * h];
        for (int y = 0; y < h; y++)
        {
            int o = (y0 + y) * stride + x0 * 4;
            for (int x = 0; x < w; x++, o += 4) m[y * w + x] = Math.Min(buf[o], Math.Min(buf[o + 1], buf[o + 2]));
        }
        return new Frame { W = w, H = h, White = m };
    }
}
