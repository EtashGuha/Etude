package com.wolfram.robottools;

import java.awt.*;
import java.awt.event.*;
import java.awt.image.*;
import javax.swing.*;
import javax.swing.event.*;
import com.wolfram.jlink.*;

public class BoxInformation extends JPanel implements MouseListener, KeyListener {

	static BufferedImage screen;

	JButton button;

	JFrame frame;

	Expr expr;

	public BoxInformation() {
		super();
	}

	public void setBoxInformation(Expr e) {
		try {
			screen = screenCapture();
		} catch (AWTException awt) {
			
		}
		expr = e;
	}

	// Displays the window with the box information drawn on top.
	public void showBoxInformation() {
		frame = fullScreenWindow();
		// frame.getContentPane().setLayout(null);
		frame.getContentPane().add(this);
		frame.addMouseListener(this);
		frame.addKeyListener(this);
		frame.validate();
	}

	public static BufferedImage screenCapture() throws AWTException {
		Dimension size;
		Robot robot = new Robot();
		Toolkit tk = Toolkit.getDefaultToolkit();
		size = tk.getScreenSize();
		Rectangle bounds = new Rectangle(0, 0, (int)size.getWidth(), (int)size.getHeight());
		return robot.createScreenCapture(bounds);
	}

	public static JFrame fullScreenWindow() {
		GraphicsDevice device = null;
		GraphicsEnvironment env = GraphicsEnvironment.getLocalGraphicsEnvironment();
		device = env.getDefaultScreenDevice();
		JFrame f = new JFrame(device.getDefaultConfiguration());
		f.setUndecorated(true);
		f.setResizable(false);
		device.setFullScreenWindow(f);
		return f;
	}

	//
	// Expect 'e' to be of this form:
	//
	// { "BoxName", {{x1,y1}, {x2,y2}}, {} }
	//
	public void drawOneBox(Expr e, Graphics g) {
		Color c1 = new Color(255, 0, 0, 20);
		Color c2 = new Color(255, 0, 0);
		Color c3 = new Color(255, 255, 0, 20);

		int x1, y1, x2, y2;
		// String s;

		try {
			// s = e.part(1).asString();
			x1 = e.part(2).part(1).part(1).asInt();
			y1 = e.part(2).part(1).part(2).asInt();
			x2 = e.part(2).part(2).part(1).asInt();
			y2 = e.part(2).part(2).part(2).asInt();

			//System.out.println("{{" + x1 + ", " + y1 + "}, {" + x2 + ", " + y2 + "}}");
			if (e.part(3).part(1).stringQ()) {
				g.setColor(c3);
			} else {
				g.setColor(c1);
			}
			g.fillRect(x1, y1, x2 - x1, y2 - y1);
			g.setColor(c2);
			g.drawRect(x1, y1, x2 - x1, y2 - y1);
			// g.drawString(s,x2,y2);

		} catch (ExprFormatException efe) {
			//System.out.println("ExprFormatException at drawOneBox.");
		}
	}

	public void drawBoxes(Expr e, Graphics g) {
		//System.out.println(e.toString());
		if (e.part(1).listQ()) {
			for (int i = 1; i <= e.length(); i++) {
				drawBoxes(e.part(i), g);
			}
		} else if (e.part(1).symbolQ() && e.part(3).length() > 0) {
			drawOneBox(e, g);
			drawBoxes(e.part(3), g);
		}
	}

	public void paintComponent(Graphics g) {
		g.drawImage(screen, 0, 0, null);
		drawBoxes(expr, g);
	}

	public void mouseReleased(MouseEvent e) {

	}

	public void mousePressed(MouseEvent e) {

	}

	public void mouseClicked(MouseEvent e) {
		frame.setVisible(false);
		frame.dispose();
	}

	public void mouseEntered(MouseEvent e) {

	}

	public void mouseExited(MouseEvent e) {

	}

	public void keyPressed(KeyEvent e) {

	}

	public void keyReleased(KeyEvent e) {

	}

	public void keyTyped(KeyEvent e) {
		frame.setVisible(false);
		frame.dispose();
	}
}
