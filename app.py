from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from fpdf import FPDF
import os
import seaborn as sns
from scipy.stats import norm
import math
import io
from datetime import datetime

app = Flask(__name__)
app.secret_key = "your_secret_key_here"
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default_secret_key')

# Variabel global untuk menyimpan hasil analisis
df = pd.DataFrame()
timing_df = pd.DataFrame()
critical_path = []
critical_path_duration = 0
Z = 0  # Initialize Z-Score

# Fungsi Pembantu
def find_critical_path(G, te_dict):
    start_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    end_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]

    max_duration = 0
    critical_path = []

    for start in start_nodes:
        for end in end_nodes:
            for path in nx.all_simple_paths(G, start, end):
                duration = sum(te_dict.get(n, 0) for n in path)
                if duration > max_duration:
                    max_duration = duration
                    critical_path = path

    return critical_path, max_duration

def validate_dependencies(activities, dependency_map):
    for act, deps in dependency_map.items():
        for dep in deps:
            if dep not in activities:
                return False, f"Dependency '{dep}' for activity '{act}' is invalid (not in activity list)."
    
    G = nx.DiGraph()
    for act in activities:
        G.add_node(act)
    
    for act, deps in dependency_map.items():
        for dep in deps:
            G.add_edge(dep, act)
    
    try:
        cycle = nx.find_cycle(G, orientation="original")
        if cycle:
            cycle_str = " -> ".join([u for u, v, k in cycle] + [cycle[0][0]])
            return False, f"Cycle detected in dependencies: {cycle_str}"
    except nx.NetworkXNoCycle:
        pass
    
    return True, ""

def calculate_earliest_times(G, te_dict):
    ES = {}
    EF = {}
    for node in nx.topological_sort(G):
        if G.in_degree(node) == 0:
            ES[node] = 0
        else:
            ES[node] = max(EF[pred] for pred in G.predecessors(node))
        EF[node] = ES[node] + te_dict[node]
    return ES, EF

def calculate_latest_times(G, te_dict, project_duration):
    LF = {}
    LS = {}
    for node in reversed(list(nx.topological_sort(G))):
        if G.out_degree(node) == 0:
            LF[node] = project_duration
        else:
            LF[node] = min(LS[succ] for succ in G.successors(node))
        LS[node] = LF[node] - te_dict[node]
    return LF, LS

# Rute Utama
@app.route("/", methods=["GET", "POST"])
def index():
    global df, timing_df, critical_path, critical_path_duration, Z
    
    if request.method == "POST":
        try:
            activities = [act.strip() for act in request.form.getlist("activity[]") if act.strip()]

            if not activities:
                flash("Minimal satu aktivitas harus diisi.", "danger")
                return redirect(url_for("index"))

            if len(set(activities)) != len(activities):
                flash("Nama aktivitas harus unik, terdapat duplikat.", "danger")
                return redirect(url_for("index"))

            # Get duration input
            duration = list(map(float, request.form.getlist("duration[]")))
            optimistic = list(map(float, request.form.getlist("optimistic[]")))
            most_likely = list(map(float, request.form.getlist("most_likely[]")))
            pessimistic = list(map(float, request.form.getlist("pessimistic[]")))
            dependencies_input = request.form.getlist("dependencies[]")

            if not (len(activities) == len(duration) == len(optimistic) == len(most_likely) == len(pessimistic) == len(dependencies_input)):
                flash("Jumlah input tidak sesuai. Harap isi semua kolom.", "danger")
                return redirect(url_for("index"))

            dependency_map = {}
            for act, dep_str in zip(activities, dependencies_input):
                if dep_str.strip():
                    dependency_map[act] = [d.strip() for d in dep_str.split(",") if d.strip()]
                else:
                    dependency_map[act] = []

            valid, message = validate_dependencies(activities, dependency_map)
            if not valid:
                flash(message, "danger")
                return redirect(url_for("index"))

            for i, (o, m, p) in enumerate(zip(optimistic, most_likely, pessimistic)):
                if not (o <= m <= p):
                    flash(f"Estimasi waktu tidak valid untuk aktivitas '{activities[i]}': Optimis ≤ Paling Mungkin ≤ Pesimis.", "danger")
                    return redirect(url_for("index"))

            te = [(o + 4*m + p) / 6 for o, m, p in zip(optimistic, most_likely, pessimistic)]
            d = [(p - o) / 6 for o, p in zip(optimistic, pessimistic)]
            v = [std**2 for std in d]

            df = pd.DataFrame({
                "Aktivitas": activities,
                "Durasi": duration,  # Add duration column
                "Estimasi Optimis": optimistic,
                "Estimasi Paling Mungkin": most_likely,
                "Estimasi Pesimis": pessimistic,
                "Waktu Perkiraan (TE)": te,
                "Deviasi Standar (S)": d,
                "Varians (V)": v
            })

            # Terapkan format untuk menghindari notasi ilmiah
            pd.set_option('display.float_format', '{:,.6f}'.format)
            
            G = nx.DiGraph()
            for act in activities:
                G.add_node(act)
            for act, deps in dependency_map.items():
                for dep in deps:
                    G.add_edge(dep, act)

            te_dict = dict(zip(activities, te))
            critical_path, critical_path_duration = find_critical_path(G, te_dict)

            ES, EF = calculate_earliest_times(G, te_dict)
            LF, LS = calculate_latest_times(G, te_dict, critical_path_duration)

            timing_df = pd.DataFrame({
                "Aktivitas": activities,
                "Mulai Tercepat (ES)": [ES[act] for act in activities],
                "Selesai Tercepat (EF)": [EF[act] for act in activities],
                "Mulai Terlambat (LS)": [LS[act] for act in activities],
                "Selesai Terlambat (LF)": [LF[act] for act in activities],
                "Slack": [max(0, LS[act] - ES[act]) for act in activities]
            })

            # Menghitung total varians jalur kritis (V)
            variance_critical_path = sum(v[activities.index(act)] for act in critical_path)
            std_dev_critical_path = math.sqrt(variance_critical_path)

            target_time = float(request.form.get("target_time", 0))  # Set default target time to 0

            if std_dev_critical_path > 0:
                Z = (target_time - critical_path_duration) / std_dev_critical_path
                probability_on_time = norm.cdf(Z)
            else:
                Z = 0
                probability_on_time = 1.0 if critical_path_duration <= target_time else 0.0

            # Visualisasi jaringan PERT
            try:
                from networkx.drawing.nx_pydot import graphviz_layout
                try:
                    pos = graphviz_layout(G, prog='dot')
                except Exception:
                    pos = nx.spring_layout(G)
            except ImportError:
                pos = nx.spring_layout(G)

            plt.figure(figsize=(12, 8))
            node_colors = ['salmon' if node in critical_path else 'lightblue' for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2500, node_shape='o')

            critical_path_edges = list(zip(critical_path[:-1], critical_path[1:]))
            edge_colors = ['red' if e in critical_path_edges else 'gray' for e in G.edges()]
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrowsize=25)

            node_labels = {}
            for node in G.nodes():
                i = activities.index(node)
                o, m, p, te_val, v_val = optimistic[i], most_likely[i], pessimistic[i], te[i], v[i]
                label = f"{node}\nO={o}\nM={m}\nP={p}\nTE={te_val:.2f}\nVar={v_val:.2f}"
                node_labels[node] = label

            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold')

            edge_labels = {(dep, act): f"{te_dict.get(act,0):.1f} hari" for act, deps in dependency_map.items() for dep in deps}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue', font_size=8)

            plt.title("Diagram Jaringan PERT dengan Jalur Kritis")
            plt.axis('off')
            plt.tight_layout()
            
            # Pastikan folder static ada
            if not os.path.exists("static"):
                os.makedirs("static")
                
            network_img_path = os.path.join("static", "pert_network.png")
            plt.savefig(network_img_path)
            plt.close()

            # Grafik PERT
            plt.figure(figsize=(8, 5))
            sns.barplot(x="Aktivitas", y="Waktu Perkiraan (TE)", data=df, palette="coolwarm")
            plt.xlabel("Aktivitas")
            plt.ylabel("Waktu Perkiraan (TE)")
            plt.title("Analisis PERT")
            
            chart_img_path = os.path.join("static", "pert_chart.png")
            plt.savefig(chart_img_path)
            plt.close()

            # Mengembalikan hasil ke template dengan urutan yang diinginkan
            return render_template("index.html",
                                   table=df.to_html(classes="table table-bordered table-hover"),
                                   timing_table=timing_df.to_html(classes="table table-bordered table-hover"),
                                   image=chart_img_path,
                                   image_network=network_img_path,
                                   critical_path=critical_path,
                                   critical_path_duration=critical_path_duration,
                                   variance_critical_path=variance_critical_path,
                                   std_dev_critical_path=std_dev_critical_path,
                                   target_time=target_time,
                                   Z=Z,
                                   probability_on_time=probability_on_time,
                                   expected_time=critical_path_duration,  # Waktu yang diharapkan (Te)
                                   variance=variance_critical_path,       # Varians jalur kritis (V)
                                   std_dev=std_dev_critical_path)         # Deviasi standar (S)

        except ValueError:
            flash("Input tidak valid! Harap masukkan angka yang valid.", "danger")
            return redirect(url_for("index"))

    return render_template("index.html")

@app.route("/export_pdf")
def export_pdf():
    # Pastikan direktori static ada
    static_dir = os.path.join(os.getcwd(), "static")
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)  # Membuat folder static jika belum ada
    
    # Inisialisasi objek PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, "Hasil Analisis PERT", ln=True, align="C")
    pdf.ln(10)

    # Header dengan informasi
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Nama: Eurene M. Rawis", ln=True)
    pdf.cell(0, 10, "NIM: 230221010033", ln=True)
    pdf.cell(0, 10, "Fakultas: Teknik Sipil", ln=True)
    pdf.ln(10)

    # Informasi jalur kritis
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Jalur Kritis: {' -> '.join(critical_path)}", ln=True)  # Ganti '→' dengan '->'
    pdf.cell(0, 10, f"Durasi Jalur Kritis: {critical_path_duration:.2f} hari", ln=True)
    pdf.ln(10)

    # Tabel aktivitas
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Tabel Aktivitas", ln=True)
    pdf.ln(5)
    
    # Tambahkan tabel aktivitas dari DataFrame
    col_widths = [30, 25, 25, 25, 30, 25, 25, 25]  # Updated for duration
    headers = ["Aktivitas", "Durasi", "Optimis", "Paling Mungkin", "Pesimis", "Waktu Perkiraan", "Std Dev", "Varians"]
    
    pdf.set_font("Arial", "B", 10)
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, border=1)
    pdf.ln()
    
    pdf.set_font("Arial", "", 8)
    for _, row in df.iterrows():
        for i, col in enumerate(["Aktivitas", "Durasi", "Estimasi Optimis", "Estimasi Paling Mungkin", "Estimasi Pesimis", "Waktu Perkiraan (TE)", "Deviasi Standar (S)", "Varians (V)"]):
            pdf.cell(col_widths[i], 10, str(row[col]), border=1)
        pdf.ln()

    pdf.add_page()
    
    # Grafik PERT
    chart_path = os.path.join("static", "pert_chart.png")
    if os.path.exists(chart_path):
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Grafik Waktu Perkiraan Aktivitas", ln=True)
        pdf.image(chart_path, x=10, w=190)
        pdf.ln(10)
    
    # Diagram jaringan
    network_path = os.path.join("static", "pert_network.png")
    if os.path.exists(network_path):
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Diagram Jaringan PERT", ln=True)
        pdf.image(network_path, x=10, w=190)
    
    # Simpan PDF
    pdf_output = os.path.join(static_dir, "laporan_pert.pdf")  # Ganti 'static_dir' dengan variabel
    pdf.output(pdf_output)  # Menyimpan file PDF
    
    return send_file(pdf_output, as_attachment=True, mimetype="application/pdf")

@app.route("/download_excel")
def download_excel():
    # Buat file Excel dalam memory
    output = io.BytesIO()
    
    # Buat Excel writer
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet Aktivitas
        df.to_excel(writer, sheet_name='Aktivitas', index=False)
        
        # Sheet Waktu
        timing_df.to_excel(writer, sheet_name='Waktu', index=False)
        
        # Sheet Jalur Kritis
        critical_path_df = pd.DataFrame({
            "Informasi": ["Jalur Kritis", "Durasi Jalur Kritis", "Varians Jalur Kritis", "Deviasi Standar"],
            "Nilai": [' -> '.join(critical_path), critical_path_duration,  # Ganti '→' dengan '->'
                     sum(df[df['Aktivitas'].isin(critical_path)]['Varians (V)']),
                     math.sqrt(sum(df[df['Aktivitas'].isin(critical_path)]['Varians (V)']))]
        })
        critical_path_df.to_excel(writer, sheet_name='Jalur_Kritis', index=False)
    
    output.seek(0)
    
    # Buat nama file dengan timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hasil_pert_analysis_{timestamp}.xlsx"
    
    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))
    app.run(host="0.0.0.0", port=port)
