import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os

class PromptEditor:
    def __init__(self, master):
        self.master = master
        self.master.title("Prompt File Editor")
        self.master.geometry("1000x400")

        self.themes_dir = "themes"
        if not os.path.exists(self.themes_dir):
            os.makedirs(self.themes_dir)

        self.current_file = None

        self.create_widgets()
        self.load_themes()

    def create_widgets(self):
        # Theme selection
        self.theme_frame = ttk.Frame(self.master, padding="10")
        self.theme_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.theme_label = ttk.Label(self.theme_frame, text="Select Theme:")
        self.theme_label.grid(row=0, column=0, sticky=tk.W)

        self.theme_var = tk.StringVar()
        self.theme_combobox = ttk.Combobox(self.theme_frame, textvariable=self.theme_var)
        self.theme_combobox.grid(row=0, column=1, sticky=(tk.W, tk.E))
        self.theme_combobox.bind("<<ComboboxSelected>>", self.load_theme_content)

        self.new_theme_button = ttk.Button(self.theme_frame, text="New Theme", command=self.new_theme)
        self.new_theme_button.grid(row=0, column=2, padx=5)

        self.delete_theme_button = ttk.Button(self.theme_frame, text="Delete Theme", command=self.delete_theme)
        self.delete_theme_button.grid(row=0, column=3, padx=5)

        # Word lists
        self.lists_frame = ttk.Frame(self.master, padding="10")
        self.lists_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.categories = ["Adjectives", "Nouns", "Verbs"]
        self.list_boxes = {}
        self.entry_vars = {}

        for i, category in enumerate(self.categories):
            frame = ttk.LabelFrame(self.lists_frame, text=category, padding="5")
            frame.grid(row=0, column=i, padx=5, sticky=(tk.W, tk.E, tk.N, tk.S))

            listbox = tk.Listbox(frame, height=10)
            listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.list_boxes[category] = listbox

            var = tk.StringVar()
            entry = ttk.Entry(frame, textvariable=var)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.entry_vars[category] = var

            add_button = ttk.Button(frame, text="Add", command=lambda c=category: self.add_word(c))
            add_button.pack(side=tk.LEFT)

            remove_button = ttk.Button(frame, text="Remove", command=lambda c=category: self.remove_word(c))
            remove_button.pack(side=tk.LEFT)

        # Save button
        self.save_button = ttk.Button(self.master, text="Save", command=self.save_theme)
        self.save_button.grid(row=2, column=0, pady=10)

        # Configure grid
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(1, weight=1)
        self.lists_frame.columnconfigure(0, weight=1)
        self.lists_frame.columnconfigure(1, weight=1)
        self.lists_frame.columnconfigure(2, weight=1)

    def load_themes(self):
        themes = [f[:-4] for f in os.listdir(self.themes_dir) if f.endswith('.txt')]
        self.theme_combobox['values'] = themes

    def load_theme_content(self, event=None):
        theme = self.theme_var.get()
        if not theme:
            return

        self.current_file = os.path.join(self.themes_dir, f"{theme}.txt")
        
        for category in self.categories:
            self.list_boxes[category].delete(0, tk.END)

        with open(self.current_file, 'r') as file:
            for line in file:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    category, word = parts[0].strip(), parts[1].strip()
                    if category == "ADJ":
                        self.list_boxes["Adjectives"].insert(tk.END, word)
                    elif category == "NOUN":
                        self.list_boxes["Nouns"].insert(tk.END, word)
                    elif category == "VERB":
                        self.list_boxes["Verbs"].insert(tk.END, word)

    def new_theme(self):
        new_theme = tk.simpledialog.askstring("New Theme", "Enter the name of the new theme:")
        if new_theme:
            new_file = os.path.join(self.themes_dir, f"{new_theme}.txt")
            with open(new_file, 'w') as file:
                pass  # Create an empty file
            self.load_themes()
            self.theme_var.set(new_theme)
            self.load_theme_content()

    def delete_theme(self):
        theme = self.theme_var.get()
        if not theme:
            messagebox.showerror("Error", "No theme selected")
            return

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete the theme '{theme}'?"):
            os.remove(os.path.join(self.themes_dir, f"{theme}.txt"))
            self.load_themes()
            self.theme_var.set('')
            for category in self.categories:
                self.list_boxes[category].delete(0, tk.END)
            self.current_file = None
            messagebox.showinfo("Success", f"Theme '{theme}' deleted successfully")

    def add_word(self, category):
        word = self.entry_vars[category].get().strip()
        if word:
            self.list_boxes[category].insert(tk.END, word)
            self.entry_vars[category].set("")

    def remove_word(self, category):
        selection = self.list_boxes[category].curselection()
        if selection:
            self.list_boxes[category].delete(selection)

    def save_theme(self):
        if not self.current_file:
            messagebox.showerror("Error", "No theme selected")
            return

        with open(self.current_file, 'w') as file:
            for word in self.list_boxes["Adjectives"].get(0, tk.END):
                file.write(f"ADJ: {word}\n")
            for word in self.list_boxes["Nouns"].get(0, tk.END):
                file.write(f"NOUN: {word}\n")
            for word in self.list_boxes["Verbs"].get(0, tk.END):
                file.write(f"VERB: {word}\n")

        messagebox.showinfo("Success", "Theme saved successfully")

if __name__ == "__main__":
    root = tk.Tk()
    app = PromptEditor(root)
    root.mainloop()

# import tkinter as tk
# from tkinter import ttk, messagebox, filedialog
# import os

# class PromptEditor:
    # def __init__(self, master):
        # self.master = master
        # self.master.title("Prompt File Editor")
        # self.master.geometry("600x400")

        # self.themes_dir = "themes"
        # if not os.path.exists(self.themes_dir):
            # os.makedirs(self.themes_dir)

        # self.current_file = None

        # self.create_widgets()
        # self.load_themes()

    # def create_widgets(self):
        # # Theme selection
        # self.theme_frame = ttk.Frame(self.master, padding="10")
        # self.theme_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # self.theme_label = ttk.Label(self.theme_frame, text="Select Theme:")
        # self.theme_label.grid(row=0, column=0, sticky=tk.W)

        # self.theme_var = tk.StringVar()
        # self.theme_combobox = ttk.Combobox(self.theme_frame, textvariable=self.theme_var)
        # self.theme_combobox.grid(row=0, column=1, sticky=(tk.W, tk.E))
        # self.theme_combobox.bind("<<ComboboxSelected>>", self.load_theme_content)

        # self.new_theme_button = ttk.Button(self.theme_frame, text="New Theme", command=self.new_theme)
        # self.new_theme_button.grid(row=0, column=2, padx=5)

        # # Word lists
        # self.lists_frame = ttk.Frame(self.master, padding="10")
        # self.lists_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # self.categories = ["Adjectives", "Nouns", "Verbs"]
        # self.list_boxes = {}
        # self.entry_vars = {}

        # for i, category in enumerate(self.categories):
            # frame = ttk.LabelFrame(self.lists_frame, text=category, padding="5")
            # frame.grid(row=0, column=i, padx=5, sticky=(tk.W, tk.E, tk.N, tk.S))

            # listbox = tk.Listbox(frame, height=10)
            # listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            # self.list_boxes[category] = listbox

            # var = tk.StringVar()
            # entry = ttk.Entry(frame, textvariable=var)
            # entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            # self.entry_vars[category] = var

            # add_button = ttk.Button(frame, text="Add", command=lambda c=category: self.add_word(c))
            # add_button.pack(side=tk.LEFT)

            # remove_button = ttk.Button(frame, text="Remove", command=lambda c=category: self.remove_word(c))
            # remove_button.pack(side=tk.LEFT)

        # # Save button
        # self.save_button = ttk.Button(self.master, text="Save", command=self.save_theme)
        # self.save_button.grid(row=2, column=0, pady=10)

        # # Configure grid
        # self.master.columnconfigure(0, weight=1)
        # self.master.rowconfigure(1, weight=1)
        # self.lists_frame.columnconfigure(0, weight=1)
        # self.lists_frame.columnconfigure(1, weight=1)
        # self.lists_frame.columnconfigure(2, weight=1)

    # def load_themes(self):
        # themes = [f[:-4] for f in os.listdir(self.themes_dir) if f.endswith('.txt')]
        # self.theme_combobox['values'] = themes

    # def load_theme_content(self, event=None):
        # theme = self.theme_var.get()
        # if not theme:
            # return

        # self.current_file = os.path.join(self.themes_dir, f"{theme}.txt")
        
        # for category in self.categories:
            # self.list_boxes[category].delete(0, tk.END)

        # with open(self.current_file, 'r') as file:
            # for line in file:
                # parts = line.strip().split(':')
                # if len(parts) == 2:
                    # category, word = parts[0].strip(), parts[1].strip()
                    # if category == "ADJ":
                        # self.list_boxes["Adjectives"].insert(tk.END, word)
                    # elif category == "NOUN":
                        # self.list_boxes["Nouns"].insert(tk.END, word)
                    # elif category == "VERB":
                        # self.list_boxes["Verbs"].insert(tk.END, word)

    # def new_theme(self):
        # new_theme = tk.simpledialog.askstring("New Theme", "Enter the name of the new theme:")
        # if new_theme:
            # new_file = os.path.join(self.themes_dir, f"{new_theme}.txt")
            # with open(new_file, 'w') as file:
                # pass  # Create an empty file
            # self.load_themes()
            # self.theme_var.set(new_theme)
            # self.load_theme_content()

    # def add_word(self, category):
        # word = self.entry_vars[category].get().strip()
        # if word:
            # self.list_boxes[category].insert(tk.END, word)
            # self.entry_vars[category].set("")

    # def remove_word(self, category):
        # selection = self.list_boxes[category].curselection()
        # if selection:
            # self.list_boxes[category].delete(selection)

    # def save_theme(self):
        # if not self.current_file:
            # messagebox.showerror("Error", "No theme selected")
            # return

        # with open(self.current_file, 'w') as file:
            # for word in self.list_boxes["Adjectives"].get(0, tk.END):
                # file.write(f"ADJ: {word}\n")
            # for word in self.list_boxes["Nouns"].get(0, tk.END):
                # file.write(f"NOUN: {word}\n")
            # for word in self.list_boxes["Verbs"].get(0, tk.END):
                # file.write(f"VERB: {word}\n")

        # messagebox.showinfo("Success", "Theme saved successfully")

# if __name__ == "__main__":
    # root = tk.Tk()
    # app = PromptEditor(root)
    # root.mainloop()