"""
A short 3Blue1Brown-style explainer for *negaverse*.

Story: teaching a model which proteins interact needs good NEGATIVE examples.
Random guessing sneaks real interactions ("hidden positives") into the negatives
and poisons training. negaverse filters candidate pairs through an hourglass
(quick-reject -> score with biology rules -> AI review) to emit clean, explained
negatives.

Render (gif, captions baked in):
    manim -qm --format gif negaverse_explainer.py Negaverse
"""

from manim import *

# ---- palette (3b1b-ish: dark ground, blue/yellow accents) -------------------
BG      = "#0e1116"
BLUE    = "#4d9be6"
TEAL    = "#3fc7c0"
YELLOW  = "#f2c94c"
GREEN   = "#5bd18b"
RED     = "#ef6f6f"
GREY    = "#8b95a3"
INK     = "#e8ecf1"


def protein(pos, color=BLUE, r=0.16):
    return Dot(point=pos, radius=r, color=color).set_sheen(0.3, UL)


class Negaverse(Scene):
    def construct(self):
        self.camera.background_color = BG
        self.title()
        self.positives_negatives()
        self.hidden_positive()
        self.hourglass()
        self.rule_closeup()
        self.payoff()

    # -- helpers --------------------------------------------------------------
    def caption(self, *lines, color=GREY, size=26):
        grp = VGroup(*[Text(t, font_size=size, color=color) for t in lines])
        grp.arrange(DOWN, buff=0.18).to_edge(DOWN, buff=0.5)
        return grp

    def clear_all(self, run_time=0.6):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=run_time)

    # -- 1. title -------------------------------------------------------------
    def title(self):
        name = Text("negaverse", font_size=88, color=INK, weight=BOLD)
        sub = Text("better negatives for biology datasets",
                   font_size=30, color=BLUE)
        sub.next_to(name, DOWN, buff=0.35)
        line = Line(LEFT, RIGHT, color=BLUE).set_width(name.width * 0.9)
        line.next_to(name, DOWN, buff=0.15).set_opacity(0.0)

        self.play(Write(name), run_time=1.4)
        self.play(FadeIn(sub, shift=UP * 0.2), run_time=0.9)
        self.wait(1.0)
        self.clear_all()

    # -- 2. positives vs negatives -------------------------------------------
    def positives_negatives(self):
        head = Text("To teach a model which proteins interact,",
                    font_size=32, color=INK).to_edge(UP, buff=0.9)
        head2 = Text("you show it examples.", font_size=32, color=INK)
        head2.next_to(head, DOWN, buff=0.2)
        self.play(FadeIn(head), FadeIn(head2), run_time=0.9)

        # positive pair (edge, green)
        pA, pB = protein(LEFT * 1.1, GREEN), protein(RIGHT * 1.1, GREEN)
        pos_grp = VGroup(pA, pB).shift(LEFT * 3.3 + DOWN * 0.4)
        edge = Line(pos_grp[0].get_center(), pos_grp[1].get_center(),
                    color=GREEN, stroke_width=6)
        pos_lbl = Text("positive", font_size=26, color=GREEN)
        pos_lbl.next_to(pos_grp, UP, buff=0.5)
        pos_sub = Text("a real interacting pair", font_size=20, color=GREY)
        pos_sub.next_to(pos_grp, DOWN, buff=0.45)

        # negative pair (no edge)
        nA, nB = protein(LEFT * 1.1, BLUE), protein(RIGHT * 1.1, BLUE)
        neg_grp = VGroup(nA, nB).shift(RIGHT * 3.3 + DOWN * 0.4)
        gap = DashedLine(neg_grp[0].get_center(), neg_grp[1].get_center(),
                         color=GREY, stroke_width=3).set_opacity(0.5)
        neg_lbl = Text("negative", font_size=26, color=BLUE)
        neg_lbl.next_to(neg_grp, UP, buff=0.5)
        neg_sub = Text("a non-interacting pair", font_size=20, color=GREY)
        neg_sub.next_to(neg_grp, DOWN, buff=0.45)

        self.play(FadeIn(pos_grp), Create(edge), FadeIn(pos_lbl), run_time=0.8)
        self.play(FadeIn(neg_grp), Create(gap), FadeIn(neg_lbl), run_time=0.8)
        self.play(FadeIn(pos_sub), FadeIn(neg_sub), run_time=0.5)
        self.wait(0.6)

        cap = self.caption(
            "Positives are collected carefully by scientists.",
            "Negatives?  Usually just random guesses.", color=YELLOW, size=26)
        self.play(FadeIn(cap), run_time=0.7)
        self.wait(1.4)
        self.clear_all()

    # -- 3. the risk: hidden positive ----------------------------------------
    def hidden_positive(self):
        head = Text("The risk of guessing", font_size=34, color=INK)
        head.to_edge(UP, buff=0.9)
        self.play(FadeIn(head), run_time=0.6)

        # a little network of proteins
        import math
        pts = []
        n = 8
        for i in range(n):
            ang = TAU * i / n
            pts.append(2.2 * np.array([math.cos(ang), math.sin(ang), 0]))
        pts = [p + DOWN * 0.2 for p in pts]
        dots = VGroup(*[protein(p, BLUE) for p in pts])

        # real interactions in the network (faint green edges)
        real_edges = VGroup(
            Line(pts[0], pts[3], color=GREEN, stroke_width=3),
            Line(pts[1], pts[5], color=GREEN, stroke_width=3),
            Line(pts[2], pts[6], color=GREEN, stroke_width=3),
        ).set_opacity(0.35)

        self.play(FadeIn(dots), run_time=0.6)
        self.play(Create(real_edges), run_time=0.7)

        # pick a "random" pair, call it a negative
        a, b = pts[2], pts[6]
        pick = DashedLine(a, b, color=YELLOW, stroke_width=4)
        tag = Text('labeled "negative"', font_size=24, color=YELLOW)
        tag.to_edge(DOWN, buff=1.3)
        self.play(dots[2].animate.set_color(YELLOW),
                  dots[6].animate.set_color(YELLOW),
                  Create(pick), FadeIn(tag), run_time=0.9)
        self.wait(0.6)

        # reveal: they actually DO interact -> hidden positive
        reveal = Line(a, b, color=RED, stroke_width=7)
        warn = Text("...but they really interact  →  hidden positive",
                    font_size=26, color=RED).to_edge(DOWN, buff=0.75)
        self.play(Transform(pick, reveal), FadeOut(tag), FadeIn(warn),
                  dots[2].animate.set_color(RED),
                  dots[6].animate.set_color(RED), run_time=1.0)
        self.wait(0.5)

        stat = Text("Naive mining mislabels ~74.6% of hidden positives.",
                    font_size=28, color=RED)
        stat.to_edge(DOWN, buff=1.35)
        self.play(FadeIn(stat), run_time=0.7)
        self.wait(1.5)
        self.clear_all()

    # -- 4. the hourglass -----------------------------------------------------
    def hourglass(self):
        head = Text("negaverse filters every candidate pair",
                    font_size=32, color=INK).to_edge(UP, buff=0.7)
        self.play(FadeIn(head), run_time=0.6)

        def stage(w, label, sub, color, count):
            box = RoundedRectangle(width=w, height=1.0, corner_radius=0.18,
                                   color=color, stroke_width=4)
            box.set_fill(color, opacity=0.10)
            t = Text(label, font_size=26, color=INK, weight=BOLD)
            s = Text(sub, font_size=18, color=GREY)
            txt = VGroup(t, s).arrange(DOWN, buff=0.08).move_to(box)
            c = Text(count, font_size=22, color=color)
            c.next_to(box, RIGHT, buff=0.35)
            return VGroup(box, txt, c)

        s1 = stage(8.5, "1 · QUICK REJECT", "drop pairs already known to interact",
                   RED, "10,000")
        s2 = stage(6.6, "2 · SCORE", "network shape  +  biology rules",
                   BLUE, "2,000")
        s3 = stage(4.6, "3 · AI REVIEW", "an LLM reads only the uncertain pairs",
                   YELLOW, "50")
        col = VGroup(s1, s2, s3).arrange(DOWN, buff=0.55).shift(DOWN * 0.3)

        arr1 = Arrow(s1[0].get_bottom(), s2[0].get_top(), color=GREY,
                     buff=0.08, stroke_width=3, max_tip_length_to_length_ratio=0.15)
        arr2 = Arrow(s2[0].get_bottom(), s3[0].get_top(), color=GREY,
                     buff=0.08, stroke_width=3, max_tip_length_to_length_ratio=0.15)

        self.play(FadeIn(s1, shift=UP * 0.2), run_time=0.7)
        self.play(GrowArrow(arr1), FadeIn(s2, shift=UP * 0.2), run_time=0.7)
        self.play(GrowArrow(arr2), FadeIn(s3, shift=UP * 0.2), run_time=0.7)
        self.wait(0.8)

        # two clean outputs
        self.play(FadeOut(head), col.animate.shift(UP * 0.6),
                  FadeOut(arr1), FadeOut(arr2), run_time=0.7)
        col_all = VGroup(col)
        out_b = VGroup(
            RoundedRectangle(width=3.4, height=1.0, corner_radius=0.15,
                             color=GREEN, stroke_width=4).set_fill(GREEN, 0.10),
            Text("fair BENCHMARK set", font_size=22, color=GREEN),
        )
        out_b[1].move_to(out_b[0])
        out_t = VGroup(
            RoundedRectangle(width=3.4, height=1.0, corner_radius=0.15,
                             color=TEAL, stroke_width=4).set_fill(TEAL, 0.10),
            Text("challenging TRAINING set", font_size=21, color=TEAL),
        )
        out_t[1].move_to(out_t[0])
        outs = VGroup(out_b, out_t).arrange(RIGHT, buff=0.9)
        outs.next_to(col, DOWN, buff=0.7)
        fork_l = Arrow(s3[0].get_bottom(), out_b[0].get_top(), color=GREY,
                       buff=0.1, stroke_width=3, max_tip_length_to_length_ratio=0.2)
        fork_r = Arrow(s3[0].get_bottom(), out_t[0].get_top(), color=GREY,
                       buff=0.1, stroke_width=3, max_tip_length_to_length_ratio=0.2)
        self.play(GrowArrow(fork_l), GrowArrow(fork_r),
                  FadeIn(out_b), FadeIn(out_t), run_time=0.9)
        self.wait(1.4)
        self.clear_all()

    # -- 5. rule closeup ------------------------------------------------------
    def rule_closeup(self):
        head = Text("Biology rules are plain text — no code",
                    font_size=32, color=INK).to_edge(UP, buff=0.9)
        self.play(FadeIn(head), run_time=0.6)

        # two compartments
        nucleus = Circle(radius=1.0, color=BLUE).set_fill(BLUE, 0.10)
        nucleus.shift(LEFT * 3.2 + DOWN * 0.3)
        nlbl = Text("nucleus", font_size=22, color=BLUE).next_to(nucleus, UP, buff=0.2)
        membrane = Circle(radius=1.0, color=TEAL).set_fill(TEAL, 0.10)
        membrane.shift(RIGHT * 3.2 + DOWN * 0.3)
        mlbl = Text("membrane", font_size=22, color=TEAL).next_to(membrane, UP, buff=0.2)

        pA = protein(nucleus.get_center(), BLUE)
        pB = protein(membrane.get_center(), TEAL)

        self.play(FadeIn(nucleus), FadeIn(nlbl), FadeIn(membrane), FadeIn(mlbl),
                  FadeIn(pA), FadeIn(pB), run_time=0.8)

        cross = VGroup(
            Line(UL, DR, color=RED, stroke_width=6),
            Line(UR, DL, color=RED, stroke_width=6),
        ).set(width=0.5).move_to(DOWN * 0.3)
        cap = Text("Different part of the cell  ⇒  they can't meet  ⇒  safe negative",
                   font_size=26, color=GREEN).to_edge(DOWN, buff=1.4)
        self.play(FadeIn(cross), FadeIn(cap), run_time=0.8)

        code = Text('when: disjoint(a.compartments, b.compartments)',
                    font="Monospace", font_size=22, color=YELLOW)
        code.to_edge(DOWN, buff=0.7)
        self.play(FadeIn(code), run_time=0.7)
        self.wait(1.6)
        self.clear_all()

    # -- 6. payoff ------------------------------------------------------------
    def payoff(self):
        head = Text("The result", font_size=34, color=INK).to_edge(UP, buff=0.9)
        sub = Text("hidden positives wrongly labeled negative (HuRI)",
                   font_size=22, color=GREY).next_to(head, DOWN, buff=0.2)
        self.play(FadeIn(head), FadeIn(sub), run_time=0.6)

        base_y = -2.2
        max_h = 3.6

        def bar(x, frac, color, label, pct):
            h = max(max_h * frac, 0.12)
            rect = Rectangle(width=1.6, height=h, color=color,
                             stroke_width=0).set_fill(color, 0.9)
            rect.move_to([x, base_y + h / 2, 0])
            lab = Text(label, font_size=22, color=INK).next_to(rect, DOWN, buff=0.25)
            val = Text(pct, font_size=26, color=color).next_to(rect, UP, buff=0.2)
            return rect, lab, val

        r1, l1, v1 = bar(-2.4, 0.746, RED, "naive mining", "74.6%")
        r2, l2, v2 = bar(2.4, 0.006, GREEN, "negaverse", "0.6%")

        base = Line([-4.6, base_y, 0], [4.6, base_y, 0], color=GREY, stroke_width=2)
        self.play(Create(base), run_time=0.4)
        self.play(GrowFromEdge(r1, DOWN), FadeIn(l1), run_time=0.8)
        self.play(FadeIn(v1), run_time=0.3)
        self.play(GrowFromEdge(r2, DOWN), FadeIn(l2), run_time=0.8)
        self.play(FadeIn(v2), run_time=0.3)
        self.wait(1.0)

        self.play(FadeOut(VGroup(head, sub, r1, l1, v1, r2, l2, v2, base)),
                  run_time=0.7)
        tag = Text("Better negatives.", font_size=44, color=INK, weight=BOLD)
        tag2 = Text("Every choice explained.", font_size=44, color=BLUE, weight=BOLD)
        grp = VGroup(tag, tag2).arrange(DOWN, buff=0.3)
        self.play(Write(tag), run_time=0.8)
        self.play(FadeIn(tag2, shift=UP * 0.2), run_time=0.8)
        self.wait(1.6)
        self.play(FadeOut(grp), run_time=0.8)
