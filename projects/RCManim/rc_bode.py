from manim import *
import numpy as np

class RC_Circuit_Bode(Scene):
    def construct(self):
        # --- Parameters ---
        R = 1e3   # 1 kΩ
        C = 1e-6  # 1 µF
        f_c = 1 / (2 * np.pi * R * C)
        freqs = np.logspace(1, 6, 500)  # 10 Hz to 1 MHz
        mag_db = 20 * np.log10(1 / np.sqrt(1 + (2*np.pi*freqs*R*C)**2))
        
        # --- Title ---
        title = Text("RC Low-Pass Filter and Its Bode Magnitude Plot").scale(0.7).to_edge(UP)
        self.play(Write(title))
        
        # =====================================================
        # --- Left side: RC Circuit Diagram ---
        # =====================================================
        # Coordinates for components
        start = LEFT * 4 + UP * 1
        resistor_pos = start + RIGHT * 2
        capacitor_pos = resistor_pos + RIGHT * 2
        ground_pos = capacitor_pos + DOWN * 2
        
        # Wires
        wire1 = Line(start, resistor_pos)
        wire2 = Line(resistor_pos, capacitor_pos)
        wire3 = Line(capacitor_pos, ground_pos)
        wire4 = Line(start + DOWN * 2, ground_pos)
        vertical_source = Line(start, start + DOWN * 2)
        
        # Components
        resistor = Rectangle(width=1.0, height=0.4, color=WHITE).move_to(resistor_pos)
        resistor_label = MathTex("R").scale(0.7).next_to(resistor, UP, buff=0.1)
        
        capacitor = VGroup(
            Line(capacitor_pos + UP*0.5, capacitor_pos + DOWN*0.1),
            Line(capacitor_pos + UP*0.1, capacitor_pos + DOWN*0.5)
        )
        capacitor_label = MathTex("C").scale(0.7).next_to(capacitor, UP, buff=0.1)
        
        ground = VGroup(
            Line(ground_pos, ground_pos + DOWN*0.3),
            Line(ground_pos + DOWN*0.3, ground_pos + DOWN*0.4 + LEFT*0.3),
            Line(ground_pos + DOWN*0.3, ground_pos + DOWN*0.4 + RIGHT*0.3)
        )
        
        vs_label = MathTex("V_{in}").scale(0.7).next_to(start, LEFT)
        vc_label = MathTex("V_{out}").scale(0.7).next_to(capacitor, RIGHT)
        
        circuit = VGroup(wire1, wire2, wire3, wire4, vertical_source,
                         resistor, capacitor, ground,
                         resistor_label, capacitor_label, vs_label, vc_label)
        circuit.shift(DOWN * 1)
        
        self.play(Create(circuit))
        self.wait(1)
        
        # --- Animate sine wave current along wire1 ---
        sine_wave = always_redraw(lambda: self.get_sine_wave(
            start=start,
            end=resistor_pos,
            amp=0.2,
            freq=5
        ))
        self.add(sine_wave)
        self.wait(0.5)
        
        # =====================================================
        # --- Right side: Bode Plot ---
        # =====================================================
        axes = Axes(
            x_range=[1, 6, 1],
            y_range=[-60, 5, 10],
            x_length=5,
            y_length=3,
            axis_config={"color": WHITE},
            x_axis_config={
                "numbers_to_include": [1, 2, 3, 4, 5, 6],
                "decimal_number_config": {"num_decimal_places": 0}
            },
            y_axis_config={"numbers_to_include": [-60, -40, -20, 0]},
        ).shift(RIGHT * 3 + UP * 1)
        
        x_label = axes.get_x_axis_label("log_{10}(f / Hz)")
        y_label = axes.get_y_axis_label("Magnitude (dB)")
        labels = VGroup(x_label, y_label)
        
        self.play(Create(axes), Write(labels))
        
        # --- Dynamic curve and moving dot ---
        self.n = 1
        curve = always_redraw(lambda: axes.plot_line_graph(
            x_values=np.log10(freqs[:self.n]),
            y_values=mag_db[:self.n],
            line_color=BLUE
        ))
        dot = always_redraw(lambda: Dot(
            point=axes.c2p(np.log10(freqs[self.n-1]), mag_db[self.n-1]),
            color=YELLOW
        ))
        self.add(curve, dot)
        
        # --- Frequency sweep + sine wave speed up ---
        for i in range(1, len(freqs), 5):
            self.n = i
            sine_wave.become(self.get_sine_wave(
                start=start,
                end=resistor_pos,
                amp=0.2,
                freq=(i/len(freqs))*50 + 1  # frequency visually increases
            ))
            self.wait(0.02)
        
        # --- Mark cutoff frequency ---
        fc_line = axes.get_vertical_line(axes.c2p(np.log10(f_c), -3))
        fc_label = MathTex("f_c = 1/(2\\pi RC)").scale(0.6).next_to(fc_line, UP)
        fc_value = Text(f"{f_c:.1f} Hz").scale(0.4).next_to(fc_label, DOWN)
        self.play(Create(fc_line), Write(fc_label), Write(fc_value))
        self.wait(2)
        
    # Helper: generate a sine wave along a wire
    def get_sine_wave(self, start, end, amp=0.2, freq=5, n_points=100):
        direction = end - start
        length = np.linalg.norm(direction)
        unit_dir = direction / length
        perp_dir = np.array([-unit_dir[1], unit_dir[0], 0])
        points = [
            start + unit_dir * (i / n_points) * length
            + amp * np.sin(2 * np.pi * freq * i / n_points) * perp_dir
            for i in range(n_points)
        ]
        return VMobject(color=YELLOW).set_points_smoothly(points)
