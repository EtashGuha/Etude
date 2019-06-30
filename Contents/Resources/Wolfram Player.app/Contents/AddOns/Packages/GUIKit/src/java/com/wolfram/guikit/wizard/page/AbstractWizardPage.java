/*
 * @(#)AbstractWizardPage.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.wizard.page;

import java.awt.Container;
import java.awt.Paint;

import javax.swing.event.EventListenerList;

import com.wolfram.guikit.wizard.Wizard;
import com.wolfram.guikit.wizard.WizardUtils;
import com.wolfram.guikit.wizard.page.event.WizardPageEvent;
import com.wolfram.guikit.wizard.page.event.WizardPageListener;
import com.wolfram.guikit.wizard.ui.PageContentPanel;
import com.wolfram.guikit.wizard.ui.PageNavigationPanel;
import com.wolfram.guikit.wizard.ui.PageSideBarPanel;

/**
 * AbstractWizardPage
 *
 * @version $Revision: 1.1 $
 */
public abstract class AbstractWizardPage implements WizardPage {
  
  private Wizard wizard = null;
  
  private Container content;
  private PageContentPanel contentPanel;
  private PageNavigationPanel navigationPanel;
  
  private PageSideBarPanel sideBarPanel;
  private String sideBarTitle;
  private Object sideBarImage;
  private Paint sideBarPaint;
  private Container sideBarContent;
  
  private WizardPage nextPage;
  private WizardPage previousPage;
  
  private boolean allowBack = true;
  private boolean allowNext = true;
  private boolean allowLast = true;
  private boolean allowFinish = true;
  private boolean allowCancel = true;
  
  private boolean active = false;
  
  private int navigationMask = 0;
  
  private String title;
    
  protected EventListenerList listeners = null;
  
  public Wizard getWizard() {
    return wizard;
    }
  public void setWizard(Wizard wizard) {
    this.wizard = wizard;
    }
    
  public int getNavigationMask() {return navigationMask;}
  public void setNavigationMask(int mask) {
    navigationMask = mask;
    }
    
  public String[] getNavigationNames() {return WizardUtils.getNavigationNames(navigationMask);}
  public void setNavigationNames(String[] names) {
    setNavigationMask( WizardUtils.getNavigationMask(names));
    }
  
  public WizardPage getNextPage() {return nextPage;}
  public void setNextPage(WizardPage page) {
    nextPage = page;
    }
  public WizardPage getPreviousPage() {return previousPage;}
  public void setPreviousPage(WizardPage page) {
    previousPage = page;
    }
  
  public String getTitle() {return title;}
  public void setTitle(String string) {
    title = string;
    }
  
  public PageContentPanel getContentPanel() {
    if (contentPanel != null) return contentPanel;
    if (wizard != null && wizard.getContainer() != null && wizard.getContainer().getCurrentPage() == this)
      return wizard.getContainer().getContentPanel();
    return null;
    }
  public void setContentPanel(PageContentPanel panel) {
    contentPanel = panel;
    }
  
  // This is the area where custom user content goes within a wizard page.
  // Normally individual pages will place custom content here and share
  // a contentPanel
  public Container getContent() {return content;}
  public void setContent(Container panel) {
    content = panel;
    } 
 
  public PageNavigationPanel getNavigationPanel() {
    if (navigationPanel != null) return navigationPanel;
    if (wizard != null) return wizard.getNavigationPanel();
    return null;
    }
  public void setNavigationPanel(PageNavigationPanel panel) {
    navigationPanel = panel;
    }
  
  public PageSideBarPanel getSideBarPanel() {
    if (sideBarPanel != null) return sideBarPanel;
    if (wizard != null) return wizard.getSideBarPanel();
    return null;
    }
  public void setSideBarPanel(PageSideBarPanel panel) {
    sideBarPanel = panel;
    }
  
  public String getSideBarTitle() {
    return sideBarTitle;
    }
  public void setSideBarTitle(String t) {
    sideBarTitle = t;
    }
  
  public Object getSideBarImage() {
    return sideBarImage;
    }
  public void setSideBarImage(Object image) {
    sideBarImage = image;
    }
  
  public Paint getSideBarPaint() {
    return sideBarPaint;
    }
  public void setSideBarPaint(Paint p) {
    sideBarPaint = p;
    }
    
  public Container getSideBarContent() {
    return sideBarContent;
    }
  public void setSideBarContent(Container c) {
    sideBarContent = c;
    }
  
  
  public boolean getAllowBack(){return allowBack;}
  public void setAllowBack(boolean val){
    allowBack = val;
    if (getWizard() != null) wizard.setAllowBack(this, allowBack);
    }
  public boolean getAllowNext(){return allowNext;}
  public void setAllowNext(boolean val){
    allowNext = val;
    if (getWizard() != null) wizard.setAllowNext(this, allowNext);
    }
  public boolean getAllowFinish(){return allowFinish;}
  public void setAllowFinish(boolean val){
    allowFinish = val;
    if (getWizard() != null) wizard.setAllowFinish(this, allowFinish);
    }
  public boolean getAllowLast(){return allowLast;}
  public void setAllowLast(boolean val){
    allowLast = val;
    if (getWizard() != null) wizard.setAllowLast(this, allowLast);
    }
  public boolean getAllowCancel(){return allowCancel;}
  public void setAllowCancel(boolean val){
    allowCancel = val;
    if (getWizard() != null) wizard.setAllowCancel(this, allowCancel);
    }
  
  public boolean isActive() {return active;}
  public void setActive(boolean val) {
    boolean oldVal = active;
    active = val;
    if (active != oldVal) {
      if (active) fireDidActivate();
      else fireDidDeactivate();
      }
    }
  
  public void fireDidActivate() {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();
    WizardPageEvent e = null;
    // Process the listeners last to first, notifying
    // those that are interested in this event
    for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
      if ( lsns[i] == WizardPageListener.class ) {
        if (e == null)
          e = new WizardPageEvent(this);
        ((WizardPageListener)lsns[i+1]).pageDidActivate(e);
        }
      }
    }
    
  public void fireWillActivate() {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();
    WizardPageEvent e = null;
    // Process the listeners last to first, notifying
    // those that are interested in this event
    for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
      if ( lsns[i] == WizardPageListener.class ) {
        if (e == null)
          e = new WizardPageEvent(this);
        ((WizardPageListener)lsns[i+1]).pageWillActivate(e);
        }
      }
    }
    
  public void fireWillDeactivate() {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();
    WizardPageEvent e = null;
    // Process the listeners last to first, notifying
    // those that are interested in this event
    for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
      if ( lsns[i] == WizardPageListener.class ) {
        if (e == null)
          e = new WizardPageEvent(this);
        ((WizardPageListener)lsns[i+1]).pageWillDeactivate(e);
        }
      }
    }
    
  public void fireDidDeactivate() {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();
    WizardPageEvent e = null;
    // Process the listeners last to first, notifying
    // those that are interested in this event
    for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
      if ( lsns[i] == WizardPageListener.class ) {
        if (e == null)
          e = new WizardPageEvent(this);
        ((WizardPageListener)lsns[i+1]).pageDidDeactivate(e);
        }
      }
    }
    
  /**
   * Adds the specified ParameterListener to receive ParameterEvents.
   * <p>
   * Use this method to register a ParameterListener object to receive
   * notifications when parameter changes occur
   *
   * @param l the ParameterListener to register
   * @see #removeParameterListener(ParameterListener)
   */
  public void addWizardPageListener(WizardPageListener l) {
    if (listeners == null) listeners = new EventListenerList();
    if ( l != null ) {
      listeners.add( WizardPageListener.class, l );
      }
    }

  /**
   * Removes the specified ParameterListener object so that it no longer receives
   * ParameterEvents.
   *
   * @param l the ParameterListener to register
   * @see #addParameterListener(ParameterListener)
   */
  public void removeWizardPageListener(WizardPageListener l) {
    if (listeners == null) return;
    if ( l != null ) {
      listeners.remove( WizardPageListener.class, l );
      }
    }
    
}