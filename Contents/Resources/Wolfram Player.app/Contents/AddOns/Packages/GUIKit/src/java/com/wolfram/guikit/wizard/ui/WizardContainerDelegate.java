/*
 * @(#)WizardContainerDelegate.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.wizard.ui;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Paint;
import java.awt.Point;
import java.awt.Toolkit;
import java.awt.Window;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.net.URL;

import javax.swing.AbstractAction;
import javax.swing.Action;
import javax.swing.ActionMap;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.InputMap;
import javax.swing.JButton;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.JRootPane;
import javax.swing.JSeparator;
import javax.swing.KeyStroke;
import javax.swing.RootPaneContainer;
import javax.swing.UIManager;

import com.wolfram.guikit.wizard.DefaultWizard;
import com.wolfram.guikit.wizard.Wizard;
import com.wolfram.guikit.wizard.WizardContainer;
import com.wolfram.guikit.wizard.page.DefaultWizardPage;
import com.wolfram.guikit.wizard.page.WizardPage;

/**
 * WizardContainerDelegate
 *
 * @version $Revision: 1.6 $
 */
public class WizardContainerDelegate implements WizardContainer {
  
  private Wizard wizard;
  private WizardPage currentPage;
  
  // These are the current active panels
  private PageContentPanel contentPanel;
  private PageNavigationPanel navigationPanel;
  private PageSideBarPanel sideBarPanel;
  
  private JPanel sidePane;
  private JPanel contentPane;
  private JPanel navigationPane;
  
  // These are shared panels if individual pages do not override
  private PageContentPanel sharedContentPanel;
  private PageNavigationPanel sharedNavigationPanel;
  private PageSideBarPanel sharedSideBarPanel;
  
  private NavigateAction backAction;
  private NavigateAction nextAction;
  private NavigateAction lastAction;
  private NavigateAction finishAction;
  private NavigateAction cancelAction;
  private NavigateAction closeAction;
  private NavigateAction helpAction;
  
  private JPopupMenu popup;
  private MouseListener popupListener;
  
  private static final int DEFAULTWIDTH = 600;
  private static final int DEFAULTHEIGHT = 420;
  private static final double SIDEBARFACTOR = 0.30;
  
  private WizardContainer container;
  private Window containerWindow;
  private RootPaneContainer rootPaneContainer;
  
  public WizardContainerDelegate(WizardContainer container) {
    this(container,
      (container != null && container instanceof Window) ? (Window)container : null,
      (container != null && container instanceof RootPaneContainer) ? (RootPaneContainer)container : null);
    }

  public WizardContainerDelegate(WizardContainer container, Window w, RootPaneContainer r) {
    this.container = container;
    this.containerWindow = w;
    this.rootPaneContainer = r;
    init();
    }
    
  public WizardContainer getContainer() {return container;}
  public Window getContainerWindow() {return containerWindow;}
  public RootPaneContainer getRootPaneContainer() {return rootPaneContainer;}
  
  public PageSideBarPanel getSideBarPanel() {return sideBarPanel;}
  public PageNavigationPanel getNavigationPanel() {return navigationPanel;}
  public PageContentPanel getContentPanel() {return contentPanel;}
  
  private void init() {
    
    // We might want to make this conditionally set
    if (containerWindow != null) {
      containerWindow.setSize(new Dimension(DEFAULTWIDTH, DEFAULTHEIGHT));
      
      containerWindow.addWindowListener(new WindowAdapter() {
        // This was added so when a wizard panel comes up for first page, a focus
        // exists for arrow keys and default button to work.
        public void windowActivated(WindowEvent e) {
          if (wizard != null && currentPage != null && currentPage.equals(wizard.getPage(0))) {
            JButton defaultButton = getDefaultButton();
            if (defaultButton != null) defaultButton.requestFocus();
            }
          }
        });
      }
    
    backAction = new NavigateAction("Back", Wizard.NAVIGATEBACK);
    nextAction = new NavigateAction("Next", Wizard.NAVIGATENEXT);
    lastAction = new NavigateAction("Last", Wizard.NAVIGATELAST);
    finishAction = new NavigateAction("Finish", Wizard.NAVIGATEFINISH);
    cancelAction = new NavigateAction("Cancel", Wizard.NAVIGATECANCEL);
    closeAction = new NavigateAction("Close", Wizard.NAVIGATECLOSE);
    helpAction = new NavigateAction("Help", Wizard.NAVIGATEHELP);
    // For now OS X does not make text+icon buttons of the same size
    if (!com.wolfram.jlink.Utils.isMacOSX()) {
      URL imageURL = WizardDialog.class.getClassLoader().getResource("images/wizard/Back16.gif");
      if (imageURL != null) {
        backAction.putValue(Action.SMALL_ICON, new ImageIcon(imageURL));
        }
      imageURL = WizardDialog.class.getClassLoader().getResource("images/wizard/Next16.gif");
      if (imageURL != null) {
        nextAction.putValue(Action.SMALL_ICON, new ImageIcon(imageURL));
        }
      }
    
    popup = new JPopupMenu();
    JMenuItem menuItem = new JMenuItem(backAction);
    Font popUpFont = new Font(menuItem.getFont().getName(),
      Font.PLAIN, menuItem.getFont().getSize());
    menuItem.setFont( popUpFont);
    popup.add(menuItem);
    menuItem = new JMenuItem(nextAction);
    menuItem.setFont( popUpFont);
    popup.add(menuItem);
    menuItem = new JMenuItem(lastAction);
    menuItem.setFont( popUpFont);
    popup.add(menuItem);
    popup.add(new JSeparator());
    menuItem = new JMenuItem(finishAction);
    menuItem.setFont( popUpFont);
    popup.add(menuItem);
    menuItem = new JMenuItem(cancelAction);
    menuItem.setFont( popUpFont);
    popup.add(menuItem);
    menuItem = new JMenuItem(closeAction);
    menuItem.setFont( popUpFont);
    popup.add(menuItem);
    popup.add(new JSeparator());
    menuItem = new JMenuItem(helpAction);
    menuItem.setFont( popUpFont);
    popup.add(menuItem);
    
    popupListener = new PopupListener();
    if (containerWindow != null) {
      containerWindow.addMouseListener(popupListener);
      }
    
    if (rootPaneContainer != null) {
      JRootPane root = rootPaneContainer.getRootPane();
      InputMap inMap = root.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW);
      ActionMap actMap = root.getActionMap();
      
      if (inMap != null && actMap != null) {
        inMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_ESCAPE,0), "wizard.canceled");
        actMap.put("wizard.canceled", cancelAction);
        inMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_LEFT,0), "wizard.pageBack");
        actMap.put("wizard.pageBack", backAction);
        inMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_RIGHT,0), "wizard.pageNext");
        actMap.put("wizard.pageNext", nextAction);
        }
      }
    
    sidePane = new JPanel(new BorderLayout());
    contentPane = new JPanel(new BorderLayout());
    navigationPane = new JPanel();
    navigationPane.setLayout(new BoxLayout(navigationPane, BoxLayout.X_AXIS));
    
    sharedSideBarPanel = new PageSideBarPanel();
    sideBarPanel = sharedSideBarPanel;
    
    sharedNavigationPanel = new PageNavigationPanel();
    navigationPanel = sharedNavigationPanel;
    sharedNavigationPanel.setNavigationMask(Wizard.NAVIGATEBACK | Wizard.NAVIGATENEXT | Wizard.NAVIGATELAST |
      Wizard.NAVIGATECANCEL | Wizard.NAVIGATEHELP);
      
    sharedContentPanel = new PageContentPanel(); 
    contentPanel = sharedContentPanel;
    if (rootPaneContainer != null)
      rootPaneContainer.getContentPane().setLayout(new BorderLayout());
    
    sidePane.setPreferredSize(new Dimension((int)(SIDEBARFACTOR*DEFAULTWIDTH), DEFAULTHEIGHT - 50));
    contentPane.setPreferredSize(new Dimension((int)((1.0-SIDEBARFACTOR)*DEFAULTWIDTH), DEFAULTHEIGHT - 50));
    
    sidePane.add(sharedSideBarPanel, BorderLayout.CENTER);
    contentPane.add(sharedContentPanel, BorderLayout.CENTER);
    navigationPane.add(Box.createHorizontalStrut((int)(SIDEBARFACTOR*DEFAULTWIDTH)));
    navigationPane.add(sharedNavigationPanel);
    
    if (rootPaneContainer != null)
      rootPaneContainer.getContentPane().add(sidePane, BorderLayout.WEST);
    
    JPanel fullNavigationPanel = new JPanel(new BorderLayout());
    fullNavigationPanel.add(new JSeparator(JSeparator.HORIZONTAL), BorderLayout.NORTH);
    fullNavigationPanel.add(navigationPane, BorderLayout.CENTER);
    
    if (rootPaneContainer != null)
      rootPaneContainer.getContentPane().add(fullNavigationPanel, BorderLayout.SOUTH);
    
    if (rootPaneContainer != null)
      rootPaneContainer.getContentPane().add(contentPane, BorderLayout.CENTER);
    
    if (containerWindow != null) {
      containerWindow.pack();
      
      if (containerWindow.getOwner() == null) {
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
        Dimension windowSize = containerWindow.getSize();
        Point centerLocation = new Point(
          (screenSize.width-windowSize.width)/2, 
          (screenSize.height-windowSize.height)/2);
        containerWindow.setLocation(centerLocation);
        }
      
      }
    
    setResizable(false);
    }
  
  public void reset() {
  	if (wizard != null) {
  		wizard.reset();
  		showPage(wizard.getPage(0));
      JButton defaultButton = getDefaultButton();
      if (defaultButton != null) defaultButton.requestFocus();
  		}
  	}
  
  public boolean isResizable() {
    if (container != null) return container.isResizable();
    else return true;
    }
  public void setResizable(boolean val) {
    if (container != null) container.setResizable(val);
    }

  // TODO think about other forms of help instead of just a modal dialog option
  
  public void showHelpDialog() {
    if (wizard != null && wizard.getHelpDialog() != null) {
      wizard.getHelpDialog().show();
      }
    }
  
  public String getTitle() {
    if (container != null) return container.getTitle();
    else return null;
    }
  
  public void setTitle(String title) {
    if (container != null)
      container.setTitle(wizard.getTitle());
    }
  
  public Wizard getWizard() {
    return wizard;
    }
  public void setWizard(Wizard w) {
    wizard = w;
    if (wizard != null) {
      wizard.setContainer(this);
      
      setTitle(wizard.getTitle());
      
      // TODO Here is where we may need to visit wizard
      // pages to determine size and possibly resize based on max page
      // or a wizard/page preferred size
      
      showPage(wizard.getPage(0));
      JButton defaultButton = getDefaultButton();
      if (defaultButton != null) defaultButton.requestFocus();
      }

    }
 
  public WizardPage getCurrentPage() {
    return currentPage;
    }
 
  protected void updatePanels() {
    updateSideBarPanel();
    updateNavigationPanel();
    updateContentPanel();
    updateNavigationActions();
    }
  
  public void remove(Component comp){
    if (container != null && container instanceof Container) 
      ((Container)container).remove(comp);
    }
  public void removeAll() {
    if (container != null && container instanceof Container) 
      ((Container)container).removeAll();
    }
  public boolean isAncestorOf(Component c) {
    if (container != null && container instanceof Container) 
      return ((Container)container).isAncestorOf(c);
    return false;
    }
    
  public void updateSideBarPanel() {
    PageSideBarPanel usePanel = null;
    boolean needsRepaint = false;
    
    if (currentPage != null) usePanel = currentPage.getSideBarPanel();
    if (usePanel == null && wizard != null) usePanel = wizard.getSideBarPanel();
    if (usePanel == null) usePanel = sharedSideBarPanel;
    
    if (sideBarPanel != usePanel && usePanel != null) {
      if (sideBarPanel != null) {
        // remove existing panel
        if (isAncestorOf(sideBarPanel)) {
          remove(sideBarPanel);
          sideBarPanel.removeMouseListener(popupListener);
          }
        needsRepaint = true;
        } 
      sideBarPanel = usePanel;
      }
          
    // add new usePanel to component tree of this
    if (sideBarPanel != null && !sidePane.isAncestorOf(sideBarPanel)) {
      sidePane.add(sideBarPanel, BorderLayout.CENTER);
      sideBarPanel.addMouseListener(popupListener);
      needsRepaint = true;
      }
        
    if (sideBarPanel != null) {
      // TODO we might want to improve the efficiency of rebuilding
      // this sideBar content each time, especially if 
      // reusing components or reusing a steps GUI
      
      Object sideBarImage = null;
      Paint sideBarPaint = null;
      Container sideBarContent = null;
      String sideBarTitle = null;
      if (currentPage != null) {
        sideBarImage = currentPage.getSideBarImage();
        sideBarContent= currentPage.getSideBarContent();
        sideBarTitle = currentPage.getSideBarTitle();
        sideBarPaint = currentPage.getSideBarPaint();
        }
      if (wizard != null) {
        if (sideBarImage == null) sideBarImage = wizard.getSideBarImage();
        if (sideBarPaint == null) sideBarPaint = wizard.getSideBarPaint();
        if (sideBarTitle == null) sideBarTitle = wizard.getSideBarTitle();
        if (sideBarContent == null) sideBarContent = wizard.getSideBarContent();
        }
        
      sideBarPanel.setSideBarTitle(sideBarTitle);
      if (sideBarPanel.getSideBarContent() != sideBarContent) needsRepaint = true;
      sideBarPanel.setSideBarContent(sideBarContent);
      if (sideBarPanel.getSideBarImage() != sideBarImage) needsRepaint = true;
      sideBarPanel.setSideBarImage(sideBarImage);
      if (sideBarPanel.getSideBarPaint() != sideBarPaint) needsRepaint = true;
      sideBarPanel.setSideBarPaint(sideBarPaint);
      }
    
    if (needsRepaint) {
      if (sideBarPanel != null) {
        sideBarPanel.doSideBarLayout();
        }
      sidePane.repaint();
      }
    }
    
  protected JButton getDefaultButton() {
    JButton defaultButton = null;
    if (navigationPanel == null || currentPage == null) return null;
    int mask = currentPage.getNavigationMask();
    if ((mask & Wizard.NAVIGATENEXT) != 0 && nextAction.isEnabled()) 
      defaultButton = navigationPanel.getNextButton();
    else if ((mask & Wizard.NAVIGATEFINISH) != 0 && finishAction.isEnabled()) 
      defaultButton = navigationPanel.getFinishButton();
    else if ((mask & Wizard.NAVIGATECLOSE) != 0 && closeAction.isEnabled()) 
      defaultButton = navigationPanel.getCloseButton();
    return defaultButton;
    }
  
  public void updateNavigationActions() {
    if (navigationPanel == null) return;
    
    if (wizard != null && currentPage != null) {
      int mask = currentPage.getNavigationMask();
      navigationPanel.setNavigationMask(mask);
      
      backAction.setEnabled(
        (!currentPage.getAllowBack() || wizard.getPreviousPage(currentPage) == null ||
         (mask & Wizard.NAVIGATEBACK) == 0) ? false : true);
      nextAction.setEnabled(
        (!currentPage.getAllowNext() || wizard.getNextPage(currentPage) == null ||
         (mask & Wizard.NAVIGATENEXT) == 0) ? false : true);
      finishAction.setEnabled(
        (!currentPage.getAllowFinish() ||
         (mask & Wizard.NAVIGATEFINISH) == 0) ? false : true);
      lastAction.setEnabled(
        (!currentPage.getAllowLast() ||
         (mask & Wizard.NAVIGATELAST) == 0) ? false : true);
      // TODO wizard level or page level logic?
      closeAction.setEnabled(
       ((mask & Wizard.NAVIGATECLOSE) == 0) ? false : true);
      cancelAction.setEnabled(
        (!currentPage.getAllowCancel() ||
          (mask & Wizard.NAVIGATECANCEL) == 0) ? false : true);
      helpAction.setEnabled(
        (wizard.getHelpDialog() == null ||
         (mask & Wizard.NAVIGATEHELP) == 0) ? false : true);
        
      JButton defaultButton = getDefaultButton();
      navigationPanel.getRootPane().setDefaultButton(defaultButton);
      if (rootPaneContainer != null) rootPaneContainer.getRootPane().setDefaultButton(defaultButton);
   
      }
    else {
      // TODO might we ever allow finish, close or cancel if no current page case?
      backAction.setEnabled(false);
      nextAction.setEnabled(false);
      finishAction.setEnabled(false);
      lastAction.setEnabled(false);
      closeAction.setEnabled(wizard != null ? false : true);
      cancelAction.setEnabled(wizard != null ? false : true);
      helpAction.setEnabled(false);
      }
    }
  
  protected void updateNavigationPanel() {
    PageNavigationPanel usePanel = null;
    if (currentPage != null) usePanel = currentPage.getNavigationPanel();
    if (usePanel == null) usePanel = sharedNavigationPanel;
    
    if (navigationPanel != usePanel && usePanel != null) {
      if (navigationPanel != null) {
        // removes existing panel
        navigationPanel.getBackButton().setAction(null);
        navigationPanel.getNextButton().setAction(null);
        navigationPanel.getLastButton().setAction(null);
        navigationPanel.getFinishButton().setAction(null);
        navigationPanel.getCancelButton().setAction(null);
        navigationPanel.getCloseButton().setAction(null);
        navigationPanel.getHelpButton().setAction(null);
        if (isAncestorOf(navigationPanel)) {
          remove(navigationPanel);
          navigationPanel.removeMouseListener(popupListener);
          }
        } 
      navigationPanel = usePanel;
      }
     
    if (navigationPanel != null && 
        backAction != navigationPanel.getBackButton().getAction()) {
      navigationPanel.getBackButton().setAction(backAction);
      navigationPanel.getNextButton().setAction(nextAction);
      navigationPanel.getLastButton().setAction(lastAction);
      navigationPanel.getFinishButton().setAction(finishAction);
      navigationPanel.getCancelButton().setAction(cancelAction);
      navigationPanel.getCloseButton().setAction(closeAction);
      navigationPanel.getHelpButton().setAction(helpAction);
      }
      
    // add new usePanel to component tree of this
    if (navigationPanel != null && !navigationPane.isAncestorOf(navigationPanel)) {
      navigationPane.removeAll();
      navigationPane.add(Box.createHorizontalStrut(DEFAULTWIDTH/3));
      navigationPane.add(navigationPanel);
      navigationPanel.addMouseListener(popupListener);
      }
       
    }
    
  public void updateContentPanel() {
    PageContentPanel usePanel = null;
    boolean needsRepaint = false;
    
    if (currentPage != null) usePanel = currentPage.getContentPanel();
    if (usePanel == null) usePanel = sharedContentPanel;
    
    if (contentPanel != usePanel && usePanel != null) {
      if (contentPanel != null) {
        // remove existing panel
        if (isAncestorOf(contentPanel)) {
          remove(contentPanel);
          contentPanel.removeMouseListener(popupListener);
          }
        needsRepaint = true;
        } 
      contentPanel = usePanel;
      }
          
    // add new usePanel to component tree of this
    if (contentPanel != null && !contentPane.isAncestorOf(contentPanel)) {
      contentPane.add(contentPanel, BorderLayout.CENTER);
      contentPanel.addMouseListener(popupListener);
      needsRepaint = true;
      }
    
    // Now need to update the contents of the ContentPanel
    if (contentPanel != null && currentPage != null) {
      contentPanel.setContent( currentPage.getContent());
      contentPanel.setTitle( currentPage.getTitle());
      needsRepaint = true;
      }
      
    if (needsRepaint) contentPane.repaint();
    }
    
  public void navigate(ActionEvent e, int type) {
    switch (type) {
      case Wizard.NAVIGATEBACK:
        if (currentPage != null && currentPage.getAllowBack()) {
          if (currentPage.getPreviousPage() != null) {
            // TODO If the page has this property set we should
            // probably set the next property on the previous
            // page to go next but perhaps only if it is null
            showPage(currentPage.getPreviousPage());
            }
          else if (wizard != null && wizard.getPreviousPage(currentPage) != null)
            showPage(wizard.getPreviousPage(currentPage));
          }
        break;
      case Wizard.NAVIGATENEXT:
        if (currentPage != null && currentPage.getAllowNext()) {
          if (currentPage.getNextPage() != null) {
            // TODO If the page has this property set we should
            // probably set the previous property on the next
            // page to go back but perhaps only if it is null
            showPage(currentPage.getNextPage());
            }
          else if (wizard != null && wizard.getNextPage(currentPage) != null)
            showPage(wizard.getNextPage(currentPage));
          }
        break;
      case Wizard.NAVIGATELAST:
        if (wizard != null && currentPage != null && currentPage.getAllowLast() &&
            wizard.getLastPage() != null) {
          showPage(wizard.getLastPage());
          }
        break;
      case Wizard.NAVIGATEFINISH:
        // TODO might we ever allow finish if no current page case?
        // This probably shouldn't be possible
        if (wizard != null && currentPage != null && currentPage.getAllowFinish()) {
          wizard.finish();
          // If there is a page after finish (possible summary page)
          // go there, otherwise close wizard
          WizardPage nextPage = null;
          if (currentPage != null && currentPage.getAllowNext()) {
            if (currentPage.getNextPage() != null)
              nextPage = currentPage.getNextPage();
            else if (wizard != null && wizard.getNextPage(currentPage) != null)
              nextPage = wizard.getNextPage(currentPage);
            }
          if (nextPage != null) showPage(nextPage);
          else performClose();
          }
        break;
      case Wizard.NAVIGATECLOSE:
        performClose();
        break;
      case Wizard.NAVIGATECANCEL:
        if (wizard != null) wizard.cancel();
        performClose();
        break;
      case Wizard.NAVIGATEHELP:
        showHelpDialog();
        break;
      }
    
    }
    
  public void performClose() {
    performClose(true);
    }
    
  public void performClose(boolean needsDispose) {
    if (wizard != null) wizard.close();
    if (needsDispose && containerWindow != null) containerWindow.dispose();
    }
  
  public void showPage(WizardPage page) {
    if (page == null || page == currentPage) return;
    
    if (page.getWizard() != wizard) {
      // TODO Either switch active wizard or generate error
      return;
      }
      
    if (currentPage != null) {
      // event out the existing page to let it dispose or stop
      currentPage.fireWillDeactivate();
      currentPage.setActive(false);
      }
    
    // Do we always store history back into each page, when
    // originally null?, possibly.
    // Don't want this but for undo/redo or history maybe
    //page.setPreviousPage(currentPage);

    page.fireWillActivate();
    currentPage = page;
    updatePanels();
    currentPage.setActive(true);
    }
  
  // This exists only for debugging purposes
  public static void main(String[] args) {
    try {
      UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
    }
    catch (Exception e){}
    
    //WizardDialog dialog = new WizardDialog();
    WizardFrame dialog = new WizardFrame();
    Wizard wizard = new DefaultWizard();
    wizard.setTitle("Demo Wizard");
    for (int i=0; i < 5; ++i) {
      WizardPage page = new DefaultWizardPage();
      page.setTitle("Demo WizardPage " + (i+1));
      if (i == 0) {
        page.setContent(new JButton("Test"));
        page.setSideBarContent(new JLabel("Help for page 1"));
        }
      wizard.addPage(page);
      }
    dialog.setWizard(wizard);
    
    dialog.addWindowListener(new WindowAdapter() {
      public void windowClosing(WindowEvent e) {
        System.exit(0);
        }
      });
    dialog.setLocationRelativeTo(null);
    dialog.show();
    }
    
  class PopupListener extends MouseAdapter {
    public void mousePressed(MouseEvent e) {
      maybeShowPopup(e);
      }
    public void mouseReleased(MouseEvent e) {
      maybeShowPopup(e);
      }
    private void maybeShowPopup(MouseEvent e) {
      if (e.isPopupTrigger()) {
        popup.show(e.getComponent(), e.getX(), e.getY());
        }
      }
    }
    
  class NavigateAction extends AbstractAction {
    private static final long serialVersionUID = -1287957975656787948L;
    
    private int type;
    public NavigateAction(String name, int type) {
      super(name);
      this.type = type;
      }
    public void actionPerformed(ActionEvent e) {
      navigate(e, type);
      }
    public int getType() {return type;}
    }

}
