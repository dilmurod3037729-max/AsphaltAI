using System.Runtime.InteropServices;

/// <summary>/dev/uinput orqali virtual Xbox 360 gamepad. Oldindan: adb shell su -c "chmod 666 /dev/uinput; setenforce 0".</summary>
public sealed class Pad : IDisposable
{
    [DllImport("libc", SetLastError = true)] static extern int open(string path, int flags);
    [DllImport("libc", SetLastError = true)] static extern int ioctl(int fd, nuint req, nint arg);
    [DllImport("libc", SetLastError = true)] static extern nint write(int fd, byte[] buf, nuint n);
    [DllImport("libc")] static extern int close(int fd);

    const int EV_SYN = 0, EV_KEY = 1, EV_ABS = 3;
    const nuint UI_SET_EVBIT = 0x40045564, UI_SET_KEYBIT = 0x40045565, UI_SET_ABSBIT = 0x40045567, UI_DEV_CREATE = 0x5501, UI_DEV_DESTROY = 0x5502;

    public static readonly Dictionary<string, int> Buttons = new()
    {
        ["A"] = 0x130, ["B"] = 0x131, ["X"] = 0x133, ["Y"] = 0x134, ["LB"] = 0x136, ["RB"] = 0x137,
        ["BACK"] = 0x13a, ["START"] = 0x13b, ["MODE"] = 0x13c, ["LS"] = 0x13d, ["RS"] = 0x13e,
    };
    // stiklar -32768..32767, triggerlar 0..255, hat -1..1
    public static readonly Dictionary<string, (int code, int min, int max)> Axes = new()
    {
        ["X"] = (0, -32768, 32767), ["Y"] = (1, -32768, 32767), ["RX"] = (3, -32768, 32767), ["RY"] = (4, -32768, 32767),
        ["LT"] = (2, 0, 255), ["RT"] = (5, 0, 255), ["HX"] = (0x10, -1, 1), ["HY"] = (0x11, -1, 1),
    };

    int fd = -1;
    readonly object gate = new();
    public bool Created => fd >= 0;

    public string Create()
    {
        fd = open("/dev/uinput", 0x1 | 0x800); // O_WRONLY | O_NONBLOCK
        if (fd < 0) return $"open /dev/uinput xato errno={Marshal.GetLastWin32Error()}";
        int bad = 0, err = 0;
        void Set(nuint req, int v) { if (ioctl(fd, req, (nint)v) < 0) { bad++; err = Marshal.GetLastWin32Error(); Android.Util.Log.Warn("AsphaltAI", $"ioctl {req:x} {v:x} errno {err}"); } }
        Set(UI_SET_EVBIT, EV_KEY); Set(UI_SET_EVBIT, EV_ABS); Set(UI_SET_EVBIT, EV_SYN);
        foreach (var c in Buttons.Values) Set(UI_SET_KEYBIT, c);
        var dev = new byte[1116];
        var name = System.Text.Encoding.ASCII.GetBytes("Microsoft X-Box 360 pad");
        Array.Copy(name, dev, name.Length);
        BitConverter.GetBytes((ushort)3).CopyTo(dev, 80);       // BUS_USB
        BitConverter.GetBytes((ushort)0x045e).CopyTo(dev, 82);  // vendor
        BitConverter.GetBytes((ushort)0x028e).CopyTo(dev, 84);  // product
        BitConverter.GetBytes((ushort)0x0110).CopyTo(dev, 86);
        foreach (var (code, min, max) in Axes.Values)
        {
            Set(UI_SET_ABSBIT, code);
            BitConverter.GetBytes(max).CopyTo(dev, 92 + 4 * code);
            BitConverter.GetBytes(min).CopyTo(dev, 92 + 256 + 4 * code);
            if (max > 255) BitConverter.GetBytes(16).CopyTo(dev, 92 + 512 + 4 * code);
            if (max > 255) BitConverter.GetBytes(128).CopyTo(dev, 92 + 768 + 4 * code);
        }
        if (write(fd, dev, (nuint)dev.Length) != dev.Length) return "uinput_user_dev yozilmadi";
        if (ioctl(fd, UI_DEV_CREATE, 0) < 0) return $"UI_DEV_CREATE xato errno={Marshal.GetLastWin32Error()}";
        return bad == 0 ? "ok" : $"ok, lekin {bad} ioctl xato errno={err}";
    }

    void Emit(int type, int code, int value)
    {
        var ev = new byte[24];
        BitConverter.GetBytes((ushort)type).CopyTo(ev, 16);
        BitConverter.GetBytes((ushort)code).CopyTo(ev, 18);
        BitConverter.GetBytes(value).CopyTo(ev, 20);
        write(fd, ev, 24);
    }

    public void Button(string name, bool down)
    {
        if (fd < 0 || !Buttons.TryGetValue(name, out var c)) return;
        lock (gate) { Emit(EV_KEY, c, down ? 1 : 0); Emit(EV_SYN, 0, 0); }
    }

    /// <summary>v: stiklar/hat uchun -1..1, triggerlar uchun 0..1.</summary>
    public void Axis(string name, double v)
    {
        if (fd < 0 || !Axes.TryGetValue(name, out var a)) return;
        int val = a.min == 0 ? (int)Math.Round(Math.Clamp(v, 0, 1) * a.max)
            : a.max == 1 ? (int)Math.Round(Math.Clamp(v, -1, 1))
            : (int)Math.Round(Math.Clamp(v, -1, 1) * (v < 0 ? -a.min : a.max));
        lock (gate) { Emit(EV_ABS, a.code, val); Emit(EV_SYN, 0, 0); }
    }

    /// <summary>"btn A 1" | "axis X 0.5" | "tap A 80" (ms) | "reset"</summary>
    public string Command(string cmd)
    {
        var p = cmd.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (p.Length == 0) return "bo'sh";
        var ci = System.Globalization.CultureInfo.InvariantCulture;
        switch (p[0])
        {
            case "btn": Button(p[1].ToUpperInvariant(), p[2] != "0"); return "ok";
            case "axis": Axis(p[1].ToUpperInvariant(), double.Parse(p[2], ci)); return "ok";
            case "tap":
                Button(p[1].ToUpperInvariant(), true);
                Task.Delay(p.Length > 2 ? int.Parse(p[2]) : 80).ContinueWith(_ => Button(p[1].ToUpperInvariant(), false));
                return "ok";
            case "reset":
                foreach (var b in Buttons.Keys) Button(b, false);
                foreach (var a in Axes.Keys) Axis(a, 0);
                return "ok";
        }
        return "noma'lum: " + p[0];
    }

    public void Dispose()
    {
        if (fd < 0) return;
        ioctl(fd, UI_DEV_DESTROY, 0); close(fd); fd = -1;
    }
}
