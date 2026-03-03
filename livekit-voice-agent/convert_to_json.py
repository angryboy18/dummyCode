import os
import json

def parse_services_md(md_path, json_path):
    print(f"Reading from {md_path}")
    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found.")
        return

    services = []
    
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Split by major headings
    sections = content.split("## ")
    
    for section in sections:
        if not section.strip():
            continue
            
        lines = section.strip().split('\n')
        title = lines[0].strip()
        
        service_data = {
            "ID": title,
            "Service title": title,
            "Category": "",
            "Sub category": "",
            "Short description": "",
            "Approx time": "",
            "Basic total cost": "",
            "Full details": section.strip()
        }
        
        for line in lines[1:]:
            line = line.strip()
            if line.startswith("- Category:"):
                service_data["Category"] = line.replace("- Category:", "").strip()
            elif line.startswith("- Sub category:"):
                service_data["Sub category"] = line.replace("- Sub category:", "").strip()
            elif line.startswith("- Short description:"):
                service_data["Short description"] = line.replace("- Short description:", "").strip()
            elif line.startswith("- Approx time:"):
                service_data["Approx time"] = line.replace("- Approx time:", "").strip() + " mins"
            elif line.startswith("- Basic total cost:"):
                service_data["Basic total cost"] = "₹" + line.replace("- Basic total cost:", "").strip()
                
        services.append(service_data)

    print(f"Parsed {len(services)} services.")
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(services, f, indent=4, ensure_ascii=False)
        
    print(f"Saved to {json_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    md_file = os.path.join(current_dir, "services.md")
    json_file = os.path.join(current_dir, "services.json")
    parse_services_md(md_file, json_file)
