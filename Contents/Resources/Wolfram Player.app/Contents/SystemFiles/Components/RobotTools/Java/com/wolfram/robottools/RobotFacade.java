package com.wolfram.robottools;

import java.awt.*;
import com.wolfram.jlink.*;

// TODO: make keyType work
// TODO: threading support, so that it is interruptible

public class RobotFacade {
	
	Robot robot;
	int autoDelay, keyPressDelay, keyReleaseDelay, keyTypeDelay, mouseMoveDelay, mousePressDelay, mouseReleaseDelay, mouseWheelDelay;
	boolean autoWaitForIdle;
	
	public RobotFacade() throws AWTException {
		robot = new Robot();
		setAutoDelay(0);
		setKeyTypeDelay(0);
		setKeyPressDelay(0);
		setKeyReleaseDelay(0);
		setMouseMoveDelay(0);
		setMousePressDelay(0);
		setMouseReleaseDelay(0);
		setMouseWheelDelay(0);
	}
	
	public void setAutoDelay(int ms)
	{
		autoDelay = ms;
	}
	
	public int getAutoDelay()
	{
		return autoDelay;
	}
	
	public void setAutoWaitForIdle(boolean b)
	{
		autoWaitForIdle = b;
	}

	public boolean getAutoWaitForIdle()
	{
		return autoWaitForIdle;
	}
	
	public void setKeyPressDelay(int ms)
	{
		keyPressDelay = ms;
	}
	
	public int getKeyPressDelay(int ms)
	{
		return keyPressDelay;
	}
	
	public void setKeyReleaseDelay(int ms)
	{
		keyReleaseDelay = ms;
	}
	
	public int getKeyReleaseDelay(int ms)
	{
		return keyReleaseDelay;
	}
	
	public void setKeyTypeDelay(int ms)
	{
		keyTypeDelay = ms;
	}
	
	public int getKeyTypeDelay(int ms)
	{
		return keyTypeDelay;
	}
	
	public void setMouseMoveDelay(int ms)
	{
		mouseMoveDelay = ms;
	}
	
	public int getMouseMoveDelay(int ms)
	{
		return mouseMoveDelay;
	}
	
	public void setMousePressDelay(int ms)
	{
		mousePressDelay = ms;
	}
	
	public int getMousePressDelay(int ms)
	{
		return mousePressDelay;
	}
	
	public void setMouseReleaseDelay(int ms)
	{
		mouseReleaseDelay = ms;
	}
	
	public int getMouseReleaseDelay(int ms)
	{
		return mouseReleaseDelay;
	}
	
	public void setMouseWheelDelay(int ms)
	{
		mouseWheelDelay = ms;
	}
	
	public int getMouseWheelDelay(int ms)
	{
		return mouseWheelDelay;
	}
	
	public void delay(int ms)
	{
		
	}
	
	public void keyPress(int k)
	{
		
	}
	
	public void keyRelease(int k)
	{
		
	}
	
	
	
	public void mouseMove(Expr e)
	{
		
	}
	
	public void mouseMove(int[][] coors)
	{
		
	}
	
	public void mouseMove(int x, int y)
	{
		
	}
	
	public void mouseClick(String button)
	{
		
	}
	
	public void mouseClick(int button)
	{
		
	}
	
	public void keyType(String s)
	{
		
	}
	
	public void mousePress(int button)
	{
		
	}
	
	public void mouseRelease(int button)
	{
		
	}
	
	public void mouseWheel(int button)
	{
		
	}
	
	public void waitForIdle()
	{
		
	}

	public void execute(Expr e) {

		int x, y, ms, keycode, buttons, wheelamt;
		String head;

		for (Expr cmd : e.args()) {
			head = cmd.head().toString();
			if (head.equals("delay")) {
				try {
					ms = cmd.part(1).asInt();
					this.delay(ms);
				} catch (Exception ex) {

				}
			} else if (head.equals("keyPress")) {
				try {
					keycode = cmd.part(1).asInt();
					this.keyPress(keycode);
				} catch (Exception ex) {

				}
			} else if (head.equals("keyRelease")) {
				try {
					keycode = cmd.part(1).asInt();
					this.keyRelease(keycode);
				} catch (Exception ex) {

				}
			} else if (head.equals("mouseMove")) {
				try {
					x = cmd.part(1).asInt();
					y = cmd.part(2).asInt();
					this.mouseMove(x, y);
				} catch (Exception ex) {

				}
			} else if (head.equals("mousePress")) {
				try {
					buttons = cmd.part(1).asInt();
					this.mousePress(buttons);
				} catch (Exception ex) {

				}
			} else if (head.equals("mouseRelease")) {
				try {
					buttons = cmd.part(1).asInt();
					this.mouseRelease(buttons);
				} catch (Exception ex) {

				}
			} else if (head.equals("mouseWheel")) {
				try {
					wheelamt = cmd.part(1).asInt();
					this.mouseWheel(wheelamt);
				} catch (Exception ex) {

				}
			} else if (head.equals("waitForIdle")) {
				this.waitForIdle();
			} else {
				// handle unrecognized command
			}
		}
	}
}
