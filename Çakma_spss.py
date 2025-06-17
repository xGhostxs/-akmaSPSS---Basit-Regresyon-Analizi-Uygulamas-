import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats
from scipy.stats import t, f

class ÇakmaSPSS:
    def __init__(self, root):
        self.root = root
        self.root.title("ÇakmaSPSS")
        self.df = None
        self.beta = None
        self.features = []

        # Arayüz Elemanları
        self.upload_button = tk.Button(root, text="📁 Dosya Yükle", command=self.load_file)
        self.upload_button.pack(pady=10)

        self.manual_entry_button = tk.Button(root, text="📝 Elle Veri Gir", command=self.manual_data_entry)
        self.manual_entry_button.pack(pady=5)

        self.info_label = tk.Label(root, text="Henüz dosya yüklenmedi veya veri girilmedi.")
        self.info_label.pack()

        self.target_label = tk.Label(root, text="Hedef (Y) Değişken:")
        self.target_label.pack()
        self.target_entry = tk.Entry(root)
        self.target_entry.pack()

        self.feature_label = tk.Label(root, text="Bağımsız Değişkenler (virgülle):")
        self.feature_label.pack()
        self.feature_entry = tk.Entry(root)
        self.feature_entry.pack()

        self.run_button = tk.Button(root, text="📊 Analizi Başlat", command=self.run_analysis)
        self.run_button.pack(pady=10)

        self.new_data_label = tk.Label(root, text="Yeni veri (virgülle):")
        self.new_data_label.pack()
        self.new_data_entry = tk.Entry(root)
        self.new_data_entry.pack()

        self.predict_button = tk.Button(root, text="📈 Tahmin Yap", command=self.predict_new_data)
        self.predict_button.pack(pady=5)

        # Tazı ve kaydırılabilir alan
        text_frame = tk.Frame(root)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.result_text = tk.Text(text_frame, height=10, width=90, wrap=tk.WORD)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(text_frame, command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.result_text.config(yscrollcommand=scrollbar.set)

        # Grafiklerin konulacağı kaydırılabilir alan
        self.graph_frame = tk.Frame(root)
        self.graph_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.graph_frame)
        self.scrollbar = tk.Scrollbar(self.graph_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
    # Dosya yükleme ve okuma işlemi için kullanılan fonksiyon
    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")])
        if filepath:
            try:
                if filepath.endswith(".csv"):
                    self.df = pd.read_csv(filepath)
                else:
                    self.df = pd.read_excel(filepath)
                self.info_label.config(text=f"Yüklendi: {filepath.split('/')[-1]}")
            except Exception as e:
                messagebox.showerror("Hata", f"Dosya okunamadı: {e}")
    #Elle veri girme işlemi için kullanılan foknsiyon
    def manual_data_entry(self):
        top = tk.Toplevel(self.root)
        top.title("Elle Veri Girişi")

        tk.Label(top, text="Sütun adları (virgülle):").pack()
        col_entry = tk.Entry(top)
        col_entry.pack()

        tk.Label(top, text="Veriler (satırları ';' ile ayırın hücreleri',' ile):").pack()
        data_entry = tk.Text(top, height=10, width=50)
        data_entry.pack()
        
        def submit_manual_data():
            try:
                cols = [c.strip() for c in col_entry.get().split(',')]
                raw_rows = data_entry.get("1.0", tk.END).strip().split(';')
                rows = [[float(cell.strip()) for cell in row.strip().split(',')] for row in raw_rows if row.strip()]
                self.df = pd.DataFrame(rows, columns=cols)
                self.info_label.config(text="Elle veri girildi.")
                top.destroy()
            except Exception as e:
                messagebox.showerror("Hata", f"Veri girilirken hata oluştu: {e}")

        tk.Button(top, text="Veriyi Kaydet", command=submit_manual_data).pack(pady=5)
    # Analizin yapıldığı fonksiyonu
    def run_analysis(self):
        if self.df is None:
            messagebox.showwarning("Uyarı", "Lütfen önce veri girin veya dosya yükleyin.")
            return

        y_col = self.target_entry.get().strip()
        x_cols = [col.strip() for col in self.feature_entry.get().split(',')]
        self.features = x_cols

        try:
            y = self.df[y_col].values.reshape(-1, 1)

            X = self.df[x_cols].values
            X = np.hstack((np.ones((X.shape[0], 1)), X))  

            beta = np.linalg.inv(X.T @ X) @ X.T @ y
            self.beta = beta
            y_hat = X @ beta
            residuals = y - y_hat

            SSR = np.sum((y_hat - np.mean(y)) ** 2)
            SSE = np.sum((y - y_hat) ** 2)
            SST = np.sum((y - np.mean(y)) ** 2)

            R2 = SSR / SST
            adj_R2 = 1 - (SSE / (len(y) - len(beta))) / (SST / (len(y) - 1))

            MSE = SSE / (len(y) - len(beta))
            MSR = SSR / (len(beta) - 1)
            F_stat = MSR / MSE
            df1 = len(beta) - 1
            df2 = len(y) - len(beta)
            p_f = 1 - f.cdf(F_stat, df1, df2)

            var_beta = MSE * np.linalg.inv(X.T @ X)
            se_beta = np.sqrt(np.diag(var_beta))
            t_values = beta.flatten() / se_beta
            p_values = [2 * (1 - t.cdf(abs(t_val), df2)) for t_val in t_values]

            shapiro_stat, shapiro_p = stats.shapiro(residuals)

            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, "Katsayılar:\n")
            for name, b, se, t_val, p_val in zip(['Intercept'] + x_cols, beta.flatten(), se_beta, t_values, p_values):
                self.result_text.insert(tk.END, f"{name}: {b:.4f} (SE={se:.4f}, t={t_val:.2f}, p={p_val:.4f})\n")
            self.result_text.insert(tk.END, f"\nR²: {R2:.4f}\nAdjusted R²: {adj_R2:.4f}\n")
            self.result_text.insert(tk.END, f"\nF Testi: F={F_stat:.2f}, df=({df1}, {df2}), p={p_f:.4f}\n")

            kritik_f = f.ppf(0.95, df1, df2)
            karar = "MODEL ANLAMLI (Red bölgesi)" if F_stat > kritik_f else "MODEL ANLAMSIZ (Kabul bölgesi)"
            self.result_text.insert(tk.END, f"F Kritiği (0.05): {kritik_f:.2f} → {karar}\n")

            self.result_text.insert(tk.END, f"\nShapiro-Wilk Normallik Testi: Statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}\n")

            self.plot_graph(y, y_hat, residuals)
            self.plot_f_distribution(F_stat, kritik_f, df1, df2)

        except Exception as e: 
            messagebox.showerror("Hata", f"Analiz sırasında hata oluştu: {e}")
    # Grafik fonksiyonları
    def plot_graph(self, y, y_hat, residuals):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].scatter(y, y_hat, color='blue')
        axs[0].plot(y, y, color='red', linestyle='--')
        axs[0].set_title("Gerçek vs Tahmin")
        axs[0].set_xlabel("Gerçek Y")
        axs[0].set_ylabel("Tahmin Y")

        axs[1].scatter(range(len(residuals)), residuals, color='green')
        axs[1].axhline(0, color='red', linestyle='--')
        axs[1].set_title("Artıklar")
        axs[1].set_xlabel("Gözlem")
        axs[1].set_ylabel("Artık")

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)
    # F dağılımını ve kritik değerleri gösteren fonksiyon
    def plot_f_distribution(self, F_stat, kritik_f, df1, df2):
        x = np.linspace(0, max(10, F_stat + 2), 500)
        y = f.pdf(x, df1, df2)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, y, label='F dağılımı')
        ax.axvline(F_stat, color='red', linestyle='--', label=f'Hesaplanan F = {F_stat:.2f}')
        ax.axvline(kritik_f, color='purple', linestyle='--', label=f'Kritik F (0.05) = {kritik_f:.2f}')
        ax.fill_between(x, 0, y, where=(x >= kritik_f), color='orange', alpha=0.3, label='Red bölgesi')
        ax.set_title("F Dağılımı ve Anlamlılık Bölgeleri")
        ax.set_xlabel("F Değeri")
        ax.set_ylabel("Yoğunluk")
        ax.legend()
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)
    # Yeni veri tahmini yapan fonksiyon
    def predict_new_data(self):
        if self.beta is None or not self.features:
            messagebox.showwarning("Uyarı", "Lütfen önce bir analiz yapın.")
            return

        try: 
            new_values = [float(v.strip()) for v in self.new_data_entry.get().split(',')]
            if len(new_values) != len(self.features):
                messagebox.showwarning("Hata", f"{len(self.features)} değişken bekleniyor.")
                return

            new_x = np.array([1] + new_values).reshape(1, -1)  # 1 for intercept
            prediction = new_x @ self.beta
            self.result_text.insert(tk.END, f"\nYeni veri tahmini: {prediction[0][0]:.4f}\n")
        except Exception as e:
            messagebox.showerror("Hata", f"Tahmin yapılırken hata oluştu: {e}")

def main():
    root = tk.Tk()
    app = ÇakmaSPSS(root)
    root.mainloop()

if __name__ == "__main__":
    main()
