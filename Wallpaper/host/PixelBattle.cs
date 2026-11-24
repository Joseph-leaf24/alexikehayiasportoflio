/*
 * Pixel Battle: the pixel battle as the Windows desktop's wallpaper, with nothing else needed.
 *
 * Run PixelBattle.exe and the battle (battle/index.html, next to it) plays on the desktop behind
 * the icons, on the main screen or on every screen, until Exit in its tray menu. It's shown in
 * WebView2, the Edge engine Windows has built in (the .NET Framework it runs on is built in too).
 *
 * It goes easy on the machine: the battle sleeps (its page suspended, nothing drawn) whenever a
 * maximised or full-screen window covers its screen, while the PC is locked, and on battery if
 * that's ticked. A click on the desktop calls in a blast there; holding the button down brings a
 * barrage. Its tray menu sets the speed, the zoom, the frame rate, which screens, whether it
 * starts with Windows; the settings are kept in %LOCALAPPDATA%\PixelBattle\settings.txt.
 *
 * Built by scripts/build-exe.mjs with the C# compiler that comes with Windows (.NET Framework 4,
 * C# 5), against Microsoft's WebView2 SDK. `PixelBattle.exe --selftest` attaches, runs for a few
 * seconds, writes what it found to selftest.log next to it, and exits.
 */
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.WinForms;
using Microsoft.Win32;

namespace PixelBattle
{
    static class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            bool test = Array.IndexOf(args, "--selftest") >= 0;
            // (PixelBattle.exe --exit: the running one closes, and puts the desktop's own wallpaper back)
            if (Array.IndexOf(args, "--exit") >= 0)
            {
                EventWaitHandle quit;
                if (EventWaitHandle.TryOpenExisting(Host.QuitEvent, out quit)) quit.Set();
                return;
            }
            bool first;
            using (var only = new Mutex(true, "Local\\PixelBattleWallpaper", out first))
            {
                // (it's the wallpaper already: one's enough)
                if (!first && !test) return;
                Application.EnableVisualStyles();
                Application.SetCompatibleTextRenderingDefault(false);
                Application.Run(new Host(test));
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /* Settings                                                            */
    /* ------------------------------------------------------------------ */

    sealed class Settings
    {
        public double Speed = 2;
        public int Zoom = 75;
        public int Fps = 30;
        public bool AllScreens = false;
        public bool Clicks = true;
        public bool PauseOnBattery = true;

        static string Folder { get { return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "PixelBattle"); } }
        static string FileName { get { return Path.Combine(Folder, "settings.txt"); } }

        public static Settings Load()
        {
            var s = new Settings();
            try
            {
                if (!File.Exists(FileName)) return s;
                foreach (var line in File.ReadAllLines(FileName))
                {
                    var kv = line.Split(new[] { '=' }, 2);
                    if (kv.Length != 2) continue;
                    var k = kv[0].Trim();
                    var v = kv[1].Trim();
                    var c = CultureInfo.InvariantCulture;
                    if (k == "speed") double.TryParse(v, NumberStyles.Float, c, out s.Speed);
                    else if (k == "zoom") int.TryParse(v, NumberStyles.Integer, c, out s.Zoom);
                    else if (k == "fps") int.TryParse(v, NumberStyles.Integer, c, out s.Fps);
                    else if (k == "screens") s.AllScreens = v == "all";
                    else if (k == "clicks") s.Clicks = v != "off";
                    else if (k == "battery") s.PauseOnBattery = v != "run";
                }
            }
            catch (Exception) { }
            return s;
        }

        public void Save()
        {
            try
            {
                Directory.CreateDirectory(Folder);
                var c = CultureInfo.InvariantCulture;
                File.WriteAllLines(FileName, new[]
                {
                    "speed=" + Speed.ToString(c),
                    "zoom=" + Zoom.ToString(c),
                    "fps=" + Fps.ToString(c),
                    "screens=" + (AllScreens ? "all" : "main"),
                    "clicks=" + (Clicks ? "on" : "off"),
                    "battery=" + (PauseOnBattery ? "pause" : "run"),
                });
            }
            catch (Exception) { }
        }

        /** The settings as the battle's page takes them, in its address. */
        public string Query()
        {
            return string.Format(CultureInfo.InvariantCulture, "?speed={0}&zoom={1}&fps={2}", Speed, Zoom, Fps);
        }

        public static string DataFolder { get { return Path.Combine(Folder, "WebView2"); } }
    }

    /* ------------------------------------------------------------------ */
    /* The app: the tray icon, the walls (one per screen), keeping watch   */
    /* ------------------------------------------------------------------ */

    sealed class Host : ApplicationContext
    {
        readonly Settings settings = Settings.Load();
        readonly List<Wall> walls = new List<Wall>();
        readonly NotifyIcon tray = new NotifyIcon();
        readonly System.Windows.Forms.Timer watch = new System.Windows.Forms.Timer();
        readonly string battle = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "battle");
        readonly bool test;
        readonly StringBuilder log = new StringBuilder();
        CoreWebView2Environment env;
        MouseHook hook;
        Signals signals;
        bool paused;
        bool locked;
        ToolStripMenuItem pauseItem;

        const string RunKey = "Software\\Microsoft\\Windows\\CurrentVersion\\Run";
        public const string QuitEvent = "Local\\PixelBattleWallpaperQuit";
        EventWaitHandle quit;

        public Host(bool selftest)
        {
            test = selftest;
            tray.Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath);
            tray.Text = "Pixel Battle";
            tray.Visible = !test;
            tray.ContextMenuStrip = BuildMenu();
            watch.Interval = 1000;
            watch.Tick += delegate { Watch(); };
            SystemEvents.DisplaySettingsChanged += OnDisplays;
            SystemEvents.SessionSwitch += OnSession;
            signals = new Signals(Rebuild);
            hook = new MouseHook(OnMouse);
            if (!test)
            {
                // (asked to close by `PixelBattle.exe --exit`)
                quit = new EventWaitHandle(false, EventResetMode.AutoReset, QuitEvent);
                var ui = SynchronizationContext.Current ?? new WindowsFormsSynchronizationContext();
                var waiter = new Thread(delegate ()
                {
                    quit.WaitOne();
                    ui.Post(delegate { Finish(); }, null);
                });
                waiter.IsBackground = true;
                waiter.Start();
            }
            Start();
        }

        void Log(string line)
        {
            log.AppendLine(DateTime.Now.ToString("HH:mm:ss.fff", CultureInfo.InvariantCulture) + " " + line);
        }

        async void Start()
        {
            if (!File.Exists(Path.Combine(battle, "index.html")))
            {
                MessageBox.Show("The battle (battle\\index.html) should be next to PixelBattle.exe.", "Pixel Battle", MessageBoxButtons.OK, MessageBoxIcon.Error);
                ExitThread();
                return;
            }
            try
            {
                // (no "occluded" throttling of its own: it's always under the desktop icons; this app decides when it sleeps)
                var options = new CoreWebView2EnvironmentOptions("--disable-features=CalculateNativeWinOcclusion --disable-backgrounding-occluded-windows --disable-renderer-backgrounding");
                env = await CoreWebView2Environment.CreateAsync(null, Settings.DataFolder, options);
                Log("WebView2 " + env.BrowserVersionString);
            }
            catch (Exception e)
            {
                Log("no WebView2: " + e.Message);
                if (!test)
                    MessageBox.Show("Pixel Battle needs Microsoft Edge WebView2, which comes with Windows 11 (and with Windows 10's updates).\n\nIt can be installed from microsoft.com/edge/webview2.\n\n" + e.Message, "Pixel Battle", MessageBoxButtons.OK, MessageBoxIcon.Error);
                Finish();
                return;
            }
            Attach();
            watch.Start();
            if (test)
            {
                // (--seconds=N: how long the self-test runs for)
                int secs = 9;
                foreach (var a in Environment.GetCommandLineArgs()) if (a.StartsWith("--seconds=")) int.TryParse(a.Substring(10), out secs);
                await Task.Delay(secs * 1000);
                var ours = new HashSet<IntPtr>();
                foreach (var w in walls) ours.Add(w.Handle);
                foreach (var w in walls) Log(w.Report() + "; covered by a window now: " + Covered.Over(w.Screen, ours));
                Finish();
            }
        }

        /** A wall on each screen it's to be on, behind the desktop icons. */
        void Attach()
        {
            if (env == null) return;
            if (!Desktop.Find())
            {
                Log("no desktop window found");
                tray.ShowBalloonTip(4000, "Pixel Battle", "Couldn't find the desktop to go behind. Try Exit and run it again.", ToolTipIcon.Warning);
                return;
            }
            Log("desktop: " + Desktop.Describe());
            var screens = settings.AllScreens ? Screen.AllScreens : new[] { Screen.PrimaryScreen };
            foreach (var s in screens)
            {
                var w = new Wall(s, test ? (Action<string>)Log : null);
                walls.Add(w);
                w.Open(env, battle, settings.Query() + (test ? "&report=1" : ""));
            }
        }

        /** Start again from the desktop up: after Explorer restarts, the screens change, or a setting needs it. */
        void Rebuild()
        {
            foreach (var w in walls) w.Close();
            walls.Clear();
            Attach();
        }

        void OnDisplays(object sender, EventArgs e)
        {
            Rebuild();
        }

        void OnSession(object sender, SessionSwitchEventArgs e)
        {
            if (e.Reason == SessionSwitchReason.SessionLock || e.Reason == SessionSwitchReason.ConsoleDisconnect || e.Reason == SessionSwitchReason.RemoteDisconnect) locked = true;
            else if (e.Reason == SessionSwitchReason.SessionUnlock || e.Reason == SessionSwitchReason.ConsoleConnect || e.Reason == SessionSwitchReason.RemoteConnect) locked = false;
            Watch();
        }

        /** Once a second: is each screen's battle to be seen? If not, it sleeps. */
        void Watch()
        {
            if (env == null) return;
            if (!Desktop.Alive())
            {
                Rebuild();
                return;
            }
            bool battery = settings.PauseOnBattery && SystemInformation.PowerStatus.PowerLineStatus == PowerLineStatus.Offline;
            var ours = new HashSet<IntPtr>();
            foreach (var w in walls) if (w.IsHandleCreated) ours.Add(w.Handle);
            foreach (var w in walls)
            {
                bool covered = !test && Covered.Over(w.Screen, ours);
                w.Awake(!paused && !locked && !battery && !covered);
            }
        }

        /** A click on the desktop itself calls in a blast there (and holding the button down, a barrage). */
        void OnMouse(int msg, int x, int y)
        {
            if (!settings.Clicks) return;
            if (msg == MouseHook.Up)
            {
                foreach (var w in walls) w.Run("window.battle && battle.press(false)");
                return;
            }
            if (!Desktop.IsDesktop(Native.WindowFromPoint(new Native.POINT(x, y)))) return;
            foreach (var w in walls)
            {
                var b = w.Screen.Bounds;
                if (!b.Contains(x, y)) continue;
                w.Run(string.Format(CultureInfo.InvariantCulture, "window.battle && battle.blast({0:0.0000},{1:0.0000})", (x - b.Left) / (double)b.Width, (y - b.Top) / (double)b.Height));
            }
        }

        void SetAll(string name, double value)
        {
            foreach (var w in walls) w.Run(string.Format(CultureInfo.InvariantCulture, "window.battle && battle.set('{0}', {1})", name, value));
            settings.Save();
        }

        /* ---- The tray menu ---- */

        ContextMenuStrip BuildMenu()
        {
            var m = new ContextMenuStrip();
            // (the ticks on the on/off ones, brought up to date each time the menu opens)
            var refresh = new List<Action>();
            m.Opening += delegate { foreach (var r in refresh) r(); };
            var title = new ToolStripMenuItem("Pixel Battle");
            title.Enabled = false;
            m.Items.Add(title);
            m.Items.Add(new ToolStripSeparator());
            pauseItem = new ToolStripMenuItem("Pause", null, delegate
            {
                paused = !paused;
                pauseItem.Text = paused ? "Resume" : "Pause";
                Watch();
            });
            m.Items.Add(pauseItem);
            m.Items.Add(Choice("Speed", new[] { "1x", "2x", "4x" }, new double[] { 1, 2, 4 }, delegate { return settings.Speed; }, delegate (double v) { settings.Speed = v; SetAll("speed", v); }));
            m.Items.Add(Choice("Zoom", new[] { "100%", "75%", "50%" }, new double[] { 100, 75, 50 }, delegate { return settings.Zoom; }, delegate (double v) { settings.Zoom = (int)v; SetAll("zoom", v); }));
            m.Items.Add(Choice("Frame rate", new[] { "20 fps (lightest)", "30 fps", "60 fps (smoothest)" }, new double[] { 20, 30, 60 }, delegate { return settings.Fps; }, delegate (double v) { settings.Fps = (int)v; SetAll("fps", v); }));
            m.Items.Add(Choice("Screens", new[] { "Main screen", "Every screen" }, new double[] { 0, 1 }, delegate { return settings.AllScreens ? 1 : 0; }, delegate (double v) { settings.AllScreens = v > 0; settings.Save(); Rebuild(); }));
            m.Items.Add(Toggle(refresh, "Clicking the desktop calls in a blast", delegate { return settings.Clicks; }, delegate (bool v) { settings.Clicks = v; settings.Save(); }));
            m.Items.Add(Toggle(refresh, "Sleep on battery", delegate { return settings.PauseOnBattery; }, delegate (bool v) { settings.PauseOnBattery = v; settings.Save(); Watch(); }));
            m.Items.Add(Toggle(refresh, "Start with Windows", StartsWithWindows, SetStartWithWindows));
            m.Items.Add(new ToolStripMenuItem("New battle", null, delegate { foreach (var w in walls) w.Run("location.reload()"); }));
            m.Items.Add(new ToolStripSeparator());
            m.Items.Add(new ToolStripMenuItem("Exit", null, delegate { Finish(); }));
            return m;
        }

        static ToolStripMenuItem Choice(string text, string[] labels, double[] values, Func<double> get, Action<double> set)
        {
            var item = new ToolStripMenuItem(text);
            for (int i = 0; i < labels.Length; i++)
            {
                double v = values[i];
                var sub = new ToolStripMenuItem(labels[i]);
                sub.Click += delegate { set(v); };
                item.DropDownItems.Add(sub);
            }
            item.DropDownOpening += delegate
            {
                for (int i = 0; i < values.Length; i++) ((ToolStripMenuItem)item.DropDownItems[i]).Checked = Math.Abs(get() - values[i]) < 0.01;
            };
            return item;
        }

        static ToolStripMenuItem Toggle(List<Action> refresh, string text, Func<bool> get, Action<bool> set)
        {
            var item = new ToolStripMenuItem(text);
            item.Click += delegate { set(!get()); };
            refresh.Add(delegate { item.Checked = get(); });
            return item;
        }

        static bool StartsWithWindows()
        {
            using (var k = Registry.CurrentUser.OpenSubKey(RunKey)) return k != null && k.GetValue("PixelBattle") != null;
        }

        static void SetStartWithWindows(bool on)
        {
            using (var k = Registry.CurrentUser.CreateSubKey(RunKey))
            {
                if (on) k.SetValue("PixelBattle", "\"" + Application.ExecutablePath + "\"");
                else k.DeleteValue("PixelBattle", false);
            }
        }

        /** Everything put back: the walls gone, the desktop's own wallpaper painted again. */
        void Finish()
        {
            watch.Stop();
            SystemEvents.DisplaySettingsChanged -= OnDisplays;
            SystemEvents.SessionSwitch -= OnSession;
            if (hook != null) hook.Dispose();
            if (signals != null) signals.DestroyHandle();
            foreach (var w in walls) w.Close();
            walls.Clear();
            Desktop.Repaint();
            tray.Visible = false;
            tray.Dispose();
            if (test)
            {
                try { File.WriteAllText(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "selftest.log"), log.ToString()); }
                catch (Exception) { }
            }
            ExitThread();
        }
    }

    /* ------------------------------------------------------------------ */
    /* A wall: the battle on one screen                                     */
    /* ------------------------------------------------------------------ */

    sealed class Wall : Form
    {
        public readonly Screen Screen;
        readonly Action<string> log;
        WebView2 view;
        bool awake = true;
        bool sleeping;
        string last = "";

        public Wall(Screen screen, Action<string> log)
        {
            Screen = screen;
            this.log = log;
            FormBorderStyle = FormBorderStyle.None;
            ShowInTaskbar = false;
            StartPosition = FormStartPosition.Manual;
            BackColor = Color.FromArgb(5, 6, 15);
            Bounds = screen.Bounds;
            Text = "Pixel Battle";
        }

        protected override CreateParams CreateParams
        {
            get
            {
                var cp = base.CreateParams;
                // (a tool window that never takes the focus: not in the taskbar, not in Alt+Tab)
                cp.ExStyle |= Native.WS_EX_TOOLWINDOW | Native.WS_EX_NOACTIVATE;
                return cp;
            }
        }

        protected override bool ShowWithoutActivation { get { return true; } }

        public async void Open(CoreWebView2Environment env, string folder, string query)
        {
            try
            {
                view = new WebView2();
                view.Dock = DockStyle.Fill;
                view.DefaultBackgroundColor = BackColor;
                Controls.Add(view);
                // Behind the desktop icons first, then shown there
                Desktop.Embed(Handle, Screen.Bounds);
                Show();
                Desktop.Embed(Handle, Screen.Bounds);
                await view.EnsureCoreWebView2Async(env);
                var core = view.CoreWebView2;
                core.Settings.AreDefaultContextMenusEnabled = false;
                core.Settings.AreDevToolsEnabled = false;
                core.Settings.IsStatusBarEnabled = false;
                core.Settings.IsZoomControlEnabled = false;
                core.Settings.AreBrowserAcceleratorKeysEnabled = false;
                core.SetVirtualHostNameToFolderMapping("pixelbattle.local", folder, CoreWebView2HostResourceAccessKind.Allow);
                core.WebMessageReceived += delegate (object s, CoreWebView2WebMessageReceivedEventArgs e)
                {
                    last = e.TryGetWebMessageAsString();
                    if (log != null) log("page: " + last);
                };
                if (log != null) core.NavigationCompleted += delegate (object s, CoreWebView2NavigationCompletedEventArgs e) { log("page loaded: " + e.IsSuccess); };
                core.Navigate("https://pixelbattle.local/index.html" + query);
            }
            catch (Exception e)
            {
                if (log != null) log("wall failed: " + e.Message);
            }
        }

        public void Run(string js)
        {
            if (view != null && view.CoreWebView2 != null && awake) view.CoreWebView2.ExecuteScriptAsync(js);
        }

        /** Awake: drawn. Asleep: hidden, its page suspended (no frames, next to no CPU). */
        public async void Awake(bool on)
        {
            if (view == null || view.CoreWebView2 == null || on == awake || sleeping) return;
            awake = on;
            try
            {
                if (on)
                {
                    view.CoreWebView2.Resume();
                    view.Visible = true;
                }
                else
                {
                    view.Visible = false;
                    sleeping = true;
                    await view.CoreWebView2.TrySuspendAsync();
                    sleeping = false;
                }
            }
            catch (Exception) { sleeping = false; }
        }

        public string Report()
        {
            var parent = Native.GetParent(Handle);
            Native.RECT r;
            Native.GetWindowRect(Handle, out r);
            return string.Format(CultureInfo.InvariantCulture, "wall on {0}: parent {1}, at {2},{3} {4}x{5}, visible {6}, page says {7}",
                Screen.DeviceName, Native.ClassOf(parent), r.Left, r.Top, r.Right - r.Left, r.Bottom - r.Top, Native.IsWindowVisible(Handle), last);
        }
    }

    /* ------------------------------------------------------------------ */
    /* The desktop: the window behind its icons that the wallpaper goes in */
    /* ------------------------------------------------------------------ */

    static class Desktop
    {
        static IntPtr parent;
        static IntPtr icons;
        static bool raised;

        /**
         * Finds where a wallpaper goes. Progman is asked (message 0x052C) to split off a WorkerW
         * window behind the icons. Up to Windows 11 23H2 that's a top-level window right after the
         * one holding the icons (SHELLDLL_DefView), and the wallpaper goes in it. From 24H2 the
         * desktop is "raised": the icons and the WorkerW are both children of Progman, and the
         * wallpaper goes in Progman, between the two (over the WorkerW, under the icons).
         */
        public static bool Find()
        {
            var progman = Native.FindWindow("Progman", null);
            if (progman == IntPtr.Zero) return false;
            IntPtr result;
            Native.SendMessageTimeout(progman, 0x052C, new IntPtr(0xD), new IntPtr(0x1), 0, 1000, out result);
            Native.SendMessageTimeout(progman, 0x052C, IntPtr.Zero, IntPtr.Zero, 0, 1000, out result);
            IntPtr worker = IntPtr.Zero;
            Native.EnumWindows(delegate (IntPtr top, IntPtr p)
            {
                if (Native.FindWindowEx(top, IntPtr.Zero, "SHELLDLL_DefView", null) != IntPtr.Zero)
                {
                    var next = Native.FindWindowEx(IntPtr.Zero, top, "WorkerW", null);
                    if (next != IntPtr.Zero) worker = next;
                }
                return true;
            }, IntPtr.Zero);
            icons = Native.FindWindowEx(progman, IntPtr.Zero, "SHELLDLL_DefView", null);
            if (icons != IntPtr.Zero && Native.FindWindowEx(progman, IntPtr.Zero, "WorkerW", null) != IntPtr.Zero)
            {
                parent = progman;
                raised = true;
                return true;
            }
            if (worker != IntPtr.Zero)
            {
                parent = worker;
                raised = false;
                return true;
            }
            return false;
        }

        public static string Describe()
        {
            return (raised ? "raised (24H2 on): in Progman, under the icons" : "classic: in the WorkerW behind the icons") + ", parent " + Native.ClassOf(parent);
        }

        public static bool Alive()
        {
            return parent != IntPtr.Zero && Native.IsWindow(parent);
        }

        /** Puts a window into the desktop over `bounds` (screen coordinates), under the icons. */
        public static void Embed(IntPtr hwnd, Rectangle bounds)
        {
            if (parent == IntPtr.Zero) return;
            long style = Native.GetWindowLongPtr(hwnd, Native.GWL_STYLE).ToInt64();
            style = (style & ~(long)(Native.WS_POPUP | Native.WS_CAPTION | Native.WS_THICKFRAME | Native.WS_SYSMENU)) | Native.WS_CHILD;
            Native.SetWindowLongPtr(hwnd, Native.GWL_STYLE, new IntPtr(style));
            if (Native.GetParent(hwnd) != parent) Native.SetParent(hwnd, parent);
            var pts = new[] { new Native.POINT(bounds.Left, bounds.Top) };
            Native.MapWindowPoints(IntPtr.Zero, parent, pts, 1);
            uint flags = Native.SWP_NOACTIVATE | Native.SWP_SHOWWINDOW | Native.SWP_FRAMECHANGED;
            if (raised) Native.SetWindowPos(hwnd, icons, pts[0].X, pts[0].Y, bounds.Width, bounds.Height, flags);
            else Native.SetWindowPos(hwnd, IntPtr.Zero, pts[0].X, pts[0].Y, bounds.Width, bounds.Height, flags | Native.SWP_NOZORDER);
        }

        /** Is this window the desktop itself (its icons, or the background between them)? */
        public static bool IsDesktop(IntPtr hwnd)
        {
            var c = Native.ClassOf(hwnd);
            if (c == "SysListView32") c = Native.ClassOf(Native.GetParent(hwnd));
            return c == "SHELLDLL_DefView" || c == "WorkerW" || c == "Progman";
        }

        /** Has the desktop paint its own wallpaper again (after the battle's gone off it). */
        public static void Repaint()
        {
            var path = new StringBuilder(520);
            if (Native.SystemParametersInfo(Native.SPI_GETDESKWALLPAPER, path.Capacity, path, 0))
                Native.SystemParametersInfo(Native.SPI_SETDESKWALLPAPER, 0, path.ToString(), 0);
        }
    }

    /* ------------------------------------------------------------------ */
    /* Is the desktop covered?                                              */
    /* ------------------------------------------------------------------ */

    static class Covered
    {
        static readonly HashSet<string> Shell = new HashSet<string> { "Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd", "SHELLDLL_DefView", "Windows.UI.Core.CoreWindow" };

        /** A maximised or full-screen window over this screen (one that's really there: not minimised, hidden or cloaked). */
        public static bool Over(Screen screen, HashSet<IntPtr> ours)
        {
            bool found = false;
            var bounds = screen.Bounds;
            Native.EnumWindows(delegate (IntPtr h, IntPtr p)
            {
                if (found || ours.Contains(h) || !Native.IsWindowVisible(h) || Native.IsIconic(h)) return true;
                int cloaked;
                if (Native.DwmGetWindowAttribute(h, Native.DWMWA_CLOAKED, out cloaked, 4) == 0 && cloaked != 0) return true;
                if (Shell.Contains(Native.ClassOf(h))) return true;
                Native.RECT r;
                if (!Native.GetWindowRect(h, out r)) return true;
                var rect = Rectangle.FromLTRB(r.Left, r.Top, r.Right, r.Bottom);
                var on = Screen.FromRectangle(rect);
                if (on.DeviceName != screen.DeviceName) return true;
                if (Native.IsZoomed(h) || (rect.Left <= bounds.Left && rect.Top <= bounds.Top && rect.Right >= bounds.Right && rect.Bottom >= bounds.Bottom)) found = true;
                return true;
            }, IntPtr.Zero);
            return found;
        }
    }

    /* ------------------------------------------------------------------ */
    /* Clicks on the desktop, and Explorer restarting                       */
    /* ------------------------------------------------------------------ */

    sealed class MouseHook : IDisposable
    {
        public const int Down = 0x0201;
        public const int Up = 0x0202;
        readonly Native.HookProc proc;
        readonly IntPtr handle;
        readonly Action<int, int, int> onMouse;

        public MouseHook(Action<int, int, int> on)
        {
            onMouse = on;
            proc = Hook;
            handle = Native.SetWindowsHookEx(14, proc, Native.GetModuleHandle(null), 0);
        }

        IntPtr Hook(int code, IntPtr wParam, IntPtr lParam)
        {
            if (code >= 0)
            {
                int msg = wParam.ToInt32();
                if (msg == Down || msg == Up)
                {
                    var pt = (Native.POINT)Marshal.PtrToStructure(lParam, typeof(Native.POINT));
                    try { onMouse(msg, pt.X, pt.Y); }
                    catch (Exception) { }
                }
            }
            return Native.CallNextHookEx(handle, code, wParam, lParam);
        }

        public void Dispose()
        {
            if (handle != IntPtr.Zero) Native.UnhookWindowsHookEx(handle);
        }
    }

    /** A hidden window for the shell's "TaskbarCreated" broadcast: Explorer has restarted, so the desktop's new. */
    sealed class Signals : NativeWindow
    {
        readonly int taskbarCreated = Native.RegisterWindowMessage("TaskbarCreated");
        readonly Action onRestart;

        public Signals(Action restart)
        {
            onRestart = restart;
            CreateHandle(new CreateParams());
        }

        protected override void WndProc(ref Message m)
        {
            if (m.Msg == taskbarCreated) onRestart();
            base.WndProc(ref m);
        }
    }

    /* ------------------------------------------------------------------ */
    /* Windows                                                              */
    /* ------------------------------------------------------------------ */

    static class Native
    {
        public const int GWL_STYLE = -16;
        public const long WS_CHILD = 0x40000000L, WS_POPUP = 0x80000000L, WS_CAPTION = 0x00C00000L, WS_THICKFRAME = 0x00040000L, WS_SYSMENU = 0x00080000L;
        public const int WS_EX_TOOLWINDOW = 0x00000080, WS_EX_NOACTIVATE = 0x08000000;
        public const uint SWP_NOZORDER = 0x0004, SWP_NOACTIVATE = 0x0010, SWP_FRAMECHANGED = 0x0020, SWP_SHOWWINDOW = 0x0040;
        public const int SPI_GETDESKWALLPAPER = 0x0073, SPI_SETDESKWALLPAPER = 0x0014;
        public const int DWMWA_CLOAKED = 14;

        [StructLayout(LayoutKind.Sequential)]
        public struct POINT
        {
            public int X, Y;
            public POINT(int x, int y) { X = x; Y = y; }
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct RECT
        {
            public int Left, Top, Right, Bottom;
        }

        public delegate bool EnumWindowsProc(IntPtr hwnd, IntPtr lParam);
        public delegate IntPtr HookProc(int code, IntPtr wParam, IntPtr lParam);

        [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern IntPtr FindWindow(string cls, string name);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern IntPtr FindWindowEx(IntPtr parent, IntPtr after, string cls, string name);
        [DllImport("user32.dll")] public static extern IntPtr SendMessageTimeout(IntPtr hwnd, uint msg, IntPtr wParam, IntPtr lParam, uint flags, uint timeout, out IntPtr result);
        [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc proc, IntPtr lParam);
        [DllImport("user32.dll")] public static extern IntPtr SetParent(IntPtr child, IntPtr parent);
        [DllImport("user32.dll")] public static extern IntPtr GetParent(IntPtr hwnd);
        [DllImport("user32.dll", EntryPoint = "GetWindowLongPtrW")] public static extern IntPtr GetWindowLongPtr(IntPtr hwnd, int index);
        [DllImport("user32.dll", EntryPoint = "SetWindowLongPtrW")] public static extern IntPtr SetWindowLongPtr(IntPtr hwnd, int index, IntPtr value);
        [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr hwnd, IntPtr after, int x, int y, int w, int h, uint flags);
        [DllImport("user32.dll")] public static extern int MapWindowPoints(IntPtr from, IntPtr to, [In, Out] POINT[] points, int count);
        [DllImport("user32.dll")] public static extern bool IsWindow(IntPtr hwnd);
        [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hwnd);
        [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr hwnd);
        [DllImport("user32.dll")] public static extern bool IsZoomed(IntPtr hwnd);
        [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hwnd, out RECT rect);
        [DllImport("user32.dll")] public static extern IntPtr WindowFromPoint(POINT pt);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetClassName(IntPtr hwnd, StringBuilder name, int max);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int RegisterWindowMessage(string name);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern bool SystemParametersInfo(int action, int param, StringBuilder value, int flags);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern bool SystemParametersInfo(int action, int param, string value, int flags);
        [DllImport("user32.dll")] public static extern IntPtr SetWindowsHookEx(int id, HookProc proc, IntPtr module, uint thread);
        [DllImport("user32.dll")] public static extern bool UnhookWindowsHookEx(IntPtr hook);
        [DllImport("user32.dll")] public static extern IntPtr CallNextHookEx(IntPtr hook, int code, IntPtr wParam, IntPtr lParam);
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)] public static extern IntPtr GetModuleHandle(string name);
        [DllImport("dwmapi.dll")] public static extern int DwmGetWindowAttribute(IntPtr hwnd, int attr, out int value, int size);

        public static string ClassOf(IntPtr hwnd)
        {
            if (hwnd == IntPtr.Zero) return "";
            var sb = new StringBuilder(256);
            GetClassName(hwnd, sb, sb.Capacity);
            return sb.ToString();
        }
    }
}
